#ifndef _FLEXFLOW_KERNELS_ARRAY_SHAPE_H
#define _FLEXFLOW_KERNELS_ARRAY_SHAPE_H

#include "kernels/legion_dim.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include "utils/stack_vector/stack_vector.h"
#include <cstddef>
#include <optional>
#include <vector>
#include "kernels/array_shape.dtg.h"

namespace FlexFlow {

nonnegative_int num_dims(ArrayShape const &);

nonnegative_int get_volume(ArrayShape const &);

TensorShape get_tensor_shape(ArrayShape const &, DataType);

ArrayShape array_shape_from_tensor_dims(TensorDims const &);

} // namespace FlexFlow

#endif
