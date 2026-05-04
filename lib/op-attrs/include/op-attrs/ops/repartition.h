#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_REPARTITION_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_REPARTITION_H

#include "op-attrs/ops/repartition_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include <tl/expected.hpp>
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/operator_task_space.dtg.h"

namespace FlexFlow {

tl::expected<ParallelTensorShape, std::string>
    get_output_shape(RepartitionAttrs const &,
                     ParallelTensorShape const &input_shape);

ParallelTensorDimDegrees get_output_parallel_dim_degrees(
    RepartitionAttrs const &,
    ParallelTensorDimDegrees const &input_degrees);

OperatorTaskSpace
    get_operator_task_space(RepartitionAttrs const &attrs,
                            ParallelTensorDimDegrees const &input_degrees);

} // namespace FlexFlow

#endif
