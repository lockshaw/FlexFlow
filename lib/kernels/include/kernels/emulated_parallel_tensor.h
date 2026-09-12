#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_EMULATED_PARALLEL_TENSOR_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_EMULATED_PARALLEL_TENSOR_H

#include "kernels/emulated_parallel_tensor.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "kernels/allocation.h"

namespace FlexFlow {

EmulatedParallelTensor random_emulated_parallel_tensor_of_shape(ParallelTensorShape const &,
                                                                Allocator &allocator,
                                                                int seed);

bool emulated_parallel_tensors_are_equal(EmulatedParallelTensor const &,
                                         EmulatedParallelTensor const &);

} // namespace FlexFlow

#endif
