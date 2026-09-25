#include "task-spec/ops/impl/flat.h"
#include "kernels/flat_kernels.h"
#include "task-spec/profiling.h"

namespace FlexFlow {

static std::optional<milliseconds_t>
    forward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();
  GenericTensorAccessorR input = acc.get_tensor<Permissions::RO>(TensorSlotName::INPUT);
  GenericTensorAccessorW output = acc.get_tensor<Permissions::WO>(TensorSlotName::OUTPUT);

  return profile(flat_forward_kernel,
                 profiling,
                 kernel_device_type,
                 "[Flat] forward_time = {:.2lf}ms\n",
                 input,
                 output);
}

static std::optional<milliseconds_t>
    backward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_profiling_settings();
  DeviceType kernel_device_type = acc.get_kernel_device_type();

  GenericTensorAccessorR output_grad =
      acc.get_tensor_grad<Permissions::RO>(TensorSlotName::OUTPUT);
  GenericTensorAccessorW input_grad =
      acc.get_tensor_grad<Permissions::RW>(TensorSlotName::INPUT);

  return profile(flat_backward_kernel,
                 profiling,
                 kernel_device_type,
                 "[Flat] backward_time = {:.2lf}ms\n",
                 output_grad,
                 input_grad);
}

TaskImplFunction get_flat_fwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{forward_task_impl}};
}

TaskImplFunction get_flat_bwd_task_impl() {
  return TaskImplFunction{FwdBwdOpTaskImplFunction{backward_task_impl}};
}

}; // namespace FlexFlow
