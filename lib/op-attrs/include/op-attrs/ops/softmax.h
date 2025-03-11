#ifndef _FLEXFLOW_SOFTMAX_ATTRS_H
#define _FLEXFLOW_SOFTMAX_ATTRS_H

#include "op-attrs/ops/softmax_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include <tl/expected.hpp>

namespace FlexFlow {

tl::expected<TensorShape, std::string>
    get_output_shape(SoftmaxAttrs const &attrs, TensorShape const &input_shape);
tl::expected<ParallelTensorShape, std::string>
    get_output_shape(SoftmaxAttrs const &attrs,
                     ParallelTensorShape const &input_shape);

} // namespace FlexFlow

#endif
