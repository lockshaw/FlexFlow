/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "reverse.h"
#include "kernels/accessor.h"
#include "kernels/reverse_kernels.h"
#include "utils/nonnegative_int/nonnegative_range.h"

namespace FlexFlow {

using namespace FlexFlow::Kernels::Reverse;
using coord_t = long long;

enum Slots { INPUT, OUTPUT, ATTRS, PROFILING };

OpTaskInvocation forward(ReverseAttrs const &attrs) {
  OpTaskBinding binding;

  binding.bind_arg(PROFILING, profiling_settings());
  binding.bind_arg(ATTRS, attrs);

  binding.bind(INPUT, input_tensor(0));
  binding.bind(OUTPUT, output_tensor(0));

  return {task_id_t::REVERSE_FWD_TASK_ID, binding};
}
OpTaskInvocation backward(ReverseAttrs const &attrs) {
  OpTaskBinding binding = infer_bwd_binding(forward(attrs).binding);

  return {task_id_t::REVERSE_BWD_TASK_ID, binding};
}

static std::optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto input = acc.get_tensor<Permissions::RO>(INPUT);
  auto output = acc.get_tensor<Permissions::WO>(OUTPUT);
  auto attrs = acc.get_argument<ReverseAttrs>(ATTRS);

  nonnegative_int output_size = output.shape.get_volume();
  auto axis = attrs.axis;
  nonnegative_int in_blk_size = 1_n;
  nonnegative_int reverse_dim_size = 1_n;
  nonnegative_int num_out_blks = 1_n;
  for (nonnegative_int i : nonnegative_range(output.shape.get_dim())) {
    if (i < axis.value) {
      in_blk_size *= output.shape.at(ff_dim_t{i});
    } else if (i == axis.value) {
      reverse_dim_size = output.shape.at(ff_dim_t{i});
    } else {
      num_out_blks *= output.shape.at(ff_dim_t{i});
    }
  }

  return profile(forward_kernel,
                 profiling,
                 "[reverse] forward_time = {:.2lf}ms\n",
                 input.get_float_ptr(),
                 output.get_float_ptr(),
                 num_out_blks.unwrap_nonnegative(),
                 reverse_dim_size.unwrap_nonnegative(),
                 in_blk_size.unwrap_nonnegative(),
                 output_size.unwrap_nonnegative());
}

static std::optional<float>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  auto input_grad = acc.get_tensor_grad<Permissions::WO>(INPUT);
  auto output_grad = acc.get_tensor_grad<Permissions::RO>(OUTPUT);
  auto attrs = acc.get_argument<ReverseAttrs>(ATTRS);

  int axis = input_grad.shape.num_dims().unwrap_nonnegative() -
             attrs.axis.value.unwrap_nonnegative() - 1;
  nonnegative_int in_blk_size = 1_n;
  nonnegative_int reverse_dim_size = 1_n;
  nonnegative_int num_out_blks = 1_n;
  for (nonnegative_int i : nonnegative_range(input_grad.shape.get_dim())) {
    if (i < axis) {
      in_blk_size *= input_grad.shape.at(ff_dim_t{i});
    } else if (i == axis) {
      reverse_dim_size = input_grad.shape.at(ff_dim_t{i});
    } else {
      num_out_blks *= input_grad.shape.at(ff_dim_t{i});
    }
  }

  return profile(backward_kernel,
                 profiling,
                 "[reverse] backward_time = {:.2lf}ms\n",
                 output_grad.get_float_ptr(),
                 input_grad.get_float_ptr(),
                 num_out_blks.unwrap_nonnegative(),
                 reverse_dim_size.unwrap_nonnegative(),
                 in_blk_size.unwrap_nonnegative(),
                 input_grad.shape.get_volume().unwrap_nonnegative());
}

TaskImplFunction get_reverse_fwd_task_impl() {
  return TaskImplFunction{FwdBwdTaskImplFunction{forward_task_impl}};
}
TaskImplFunction get_reverse_bwd_task_impl() {
  return TaskImplFunction{FwdBwdTaskImplFunction{backward_task_impl}};
}

OpTaskSignature get_reverse_fwd_signature() {
  OpTaskSignature fwd(OpTaskType::FWD);

  fwd.add_arg_slot<ProfilingSettings>(PROFILING);
  fwd.add_input_slot(INPUT);
  fwd.add_output_slot(OUTPUT);
  return fwd;
}

OpTaskSignature get_reverse_bwd_signature() {
  OpTaskSignature bwd = infer_bwd_signature(get_reverse_fwd_signature());
  return bwd;
}

std::vector<task_id_t> get_task_ids(ReverseAttrs const &) {
  return {task_id_t::REVERSE_FWD_TASK_ID, task_id_t::REVERSE_BWD_TASK_ID};
}

}; // namespace FlexFlow
