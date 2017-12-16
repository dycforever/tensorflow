/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/graph/control_flow.h"

#include <deque>
#include <vector>

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/errors.h"


/*
 * from :
 *
 * So, here's what I understand so far. Perhaps you can correct any misunderstandings and also this will serve as a useful resource for the time being for anyone else who runs into this.
 *
 * 1. tf.while_loop accepts a list of loop variables, a function mapping loop variables to a boolean, and a function mapping loop variables to a new set of loop variables.
 * 2. Internally, this is represented using the special nodes Enter, Exit, NextIteration, Switch, and Merge. Enter, Exit, NextIteration are all semantically equivalent to identity ops (they just forward their input to their output, potentially as a reference), but the fact that they have type Enter, Exit, NextIteration is used by the executor to handle them in a special way. The graph is constructed as follows:
 *      The loop variables are sent through "Enter" nodes.
 *      The Enter node results are then given to "Merge" nodes. During the graph construction, the inputs to the "Merge" nodes are two copies of each enter node; when the NextIteration nodes are constructed, the Merge nodes are fixed by replacing one of the Enter inputs with a NextIteration input. In this way, every Merge node (one per variable) gets an input from its respective variable's Enter and NextIteration nodes.
 *      The output of the Merge nodes is passed to the condition function, which takes them and outputs a boolean. This boolean is passed to a LoopCond node. This boolean, as well as the output of the Merge nodes, is passed to Switch nodes, again one per variable. The Switch nodes output a dead tensor to one of their outputs and a live tensor (the merge node output) to the other one, depending on the boolean.
 *      The output of the Switch node is sent to an Exit node (one per variable) or to an Identity op (one per variable), depending on whether the loop condition is false.
 *      The identity op output is given to the loop body, and the outputs of the loop body are fed to NextIteration ops; these ops are the ones patched back in as inputs to the Merge nodes.
 * 3. The executor has special support for these five primitive ops which make this structure into a loop:
 *      The executor has a concept of a Frame, which is essentially the current iteration of the innermost loop. A frame has state, where all the input and output tensors are stored in a flat vector; each op writes its outputs to a subset of the output vector and gets inputs from a subset of the input vectors; thus, the inputs and outputs of an op can be obtained by just going to the right offset in this vector of Entry values.
 * 4. A new frame is created when the executor sees an Enter node. A frame is removed when it sees an Exit node. The next iteration of the frame is progressed to when it sees a NextIteration node.
 * 5. When it sees a NextIteration node, it finds the child of that node (namely the Merge op) and calls ActivateNode on it, in order to continue the loop. Since nodes are not marked ready until all their inputs are non-dead, the nodes that get dead inputs from Switch (e.g. the loop is done) will not get run again.
 * 6. For every loop during forward propagation, a few things have to happen to create the backprop gradient graph:
 *      First of all, a loop is added to the forward propagation which counts the number of iterations. More accurately, the original loop is added to; this works because of the way the primitive ops are designed. This loop starts with a f_count Enter node and is created in control_flow_ops.py AddForwardLoopCounter.
 *      A history_map is maintained of tensors produced during forward prop, and whenever the backprop needs a tensor from the forward prop, a stack is introduced, and the forward prop has a StackPush added to it, while the backprop has a StackPop added to it that pops from the same stack. In that manner, the forward prop pushes anything the backprop will need onto a stack, and the backprop slowly consumes that stack.
 *
 *
 * and 
 *
 * Question: 
 * 1. Why is there a LoopCond node? Why not pass the output of the condition directly to Switch?
 * 2. What was the motivation for such a seemingly complicated set of primitive ops? It seems like it's possible to build other control structures on top of them – is that the goal? Were these primitive ops chosen because they make it possible to implement the fairly complex gradient loop generation?
 * 3. What is an example usecase for parallel_iterations? (This is a simple question which might make sense to add to the tf.while_loop docs))
 *
 * Answer:
 * 1. LoopCond is just used as a unique marker so we could understand the graph structure in later stages of graph rewriting. For example, rewriting the graph to support distributed execution of a loop.
 *
 * 2. Yes, that was one of the main design considerations. If you are familiar with the dataflow machines invented in the 70s, you would not be surprised by the choice of the primitives. :-) The other main considerations are non-strictness, automatic differentiation, and parallel and distributed execution of both forward and backprop.
 *
 * 3. A simple example is tf.map. And it is one of the main reasons that the performance of dynamic_rnn can be as good as static unrolling. For example it allows the dynamic unrolling logic runs on CPU and the actual computation runs on GPU, completely in parallel.
 */

namespace tensorflow {

Status BuildControlFlowInfo(Graph* g, std::vector<ControlFlowInfo>* info) {
  info->clear();
  info->resize(g->num_node_ids());

  std::vector<const Node*> parent_nodes;
  parent_nodes.resize(g->num_node_ids());

  // dyc: src_node is _SOURCE
  Node* src_node = g->source_node();
  ControlFlowInfo& src_info = (*info)[src_node->id()];
  src_info.frame = src_node;
  src_info.parent_frame = src_node;

  string frame_name;
  std::deque<Node*> ready;
  ready.push_back(src_node);
  // dyc: 从 _SOURCE 开始根据 edge 的方向遍历所有子 Node，ready 中存放了所有待处理的 Node
  while (!ready.empty()) {
    Node* curr_node = ready.front();
    ready.pop_front();
    const ControlFlowInfo& curr_info = (*info)[curr_node->id()];
    const Node* frame = curr_info.frame;
    const Node* parent = curr_info.parent_frame;
    frame_name = curr_info.frame_name;

    if (IsExit(curr_node)) {
      // Exit to the parent frame.
      const ControlFlowInfo& parent_info = (*info)[parent->id()];
      frame = parent_info.frame;
      parent = parent_info.parent_frame;
      frame_name = parent_info.frame_name;
    }

    for (const Edge* out_edge : curr_node->out_edges()) {
      // dyc: out_edge is from curr_node->out
      Node* out = out_edge->dst();
      int out_id = out->id();
      ControlFlowInfo* out_info = &(*info)[out_id];
      const Node* out_parent = out_info->parent_frame;
      bool is_visited = (parent_nodes[out_id] != nullptr);

      // Skip Sink/Source nodes.
      if (!out->IsOp()) continue;

      // Add to ready queue if not seen.
      if (!is_visited) {
        parent_nodes[out->id()] = curr_node;
        ready.push_back(out);
      }

      // Process the node 'out'.
      if (IsEnter(out)) {
        if (is_visited) {
          const string& parent_frame = (*info)[out_parent->id()].frame_name;
          if (parent_frame != frame_name) {
            return errors::InvalidArgument(
                "The node '", out->name(),
                "' has inputs from different "
                "frames. The input '",
                curr_node->name(), "' is in frame '", frame_name,
                "'. The input '", parent_nodes[out->id()]->name(),
                "' is in frame '", parent_frame, "'.");
          }
        } else {
          out_info->frame = out;
          out_info->parent_frame = frame;
          TF_RETURN_IF_ERROR(
              GetNodeAttr(out->attrs(), "frame_name", &out_info->frame_name));
          if (out_info->frame_name.empty()) {
            return errors::InvalidArgument("The Enter node ", out->name(),
                                           " must have a frame name.");
          }
        }
      } else {
        if (is_visited) {
          if (out_info->frame_name != frame_name) {
            return errors::InvalidArgument(
                "The node '", out->name(),
                "' has inputs from different "
                "frames. The input '",
                curr_node->name(), "' is in frame '", frame_name,
                "'. The input '", parent_nodes[out->id()]->name(),
                "' is in frame '", out_info->frame_name, "'.");
          }
        } else {
          out_info->frame = frame;
          out_info->parent_frame = parent;
          out_info->frame_name = frame_name;
        }
      }
    }
  }
  return Status::OK();
}

}  // namespace tensorflow
