#ifndef _FLEXFLOW_DROPOUT_ATTRS_H
#define _FLEXFLOW_DROPOUT_ATTRS_H

#include "op-attrs/ops/dropout_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include <tl/expected.hpp>

namespace FlexFlow {

TensorShape get_output_shape(DropoutAttrs const &, TensorShape const &);
tl::expected<ParallelTensorShape, std::string>
    get_output_shape(DropoutAttrs const &, ParallelTensorShape const &);

} // namespace FlexFlow

#endif
