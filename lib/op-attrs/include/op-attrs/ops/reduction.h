#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_REDUCTION_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_REDUCTION_H

#include "op-attrs/ops/reduction_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include <tl/expected.hpp>
#include "op-attrs/operator_task_space.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"

namespace FlexFlow {

tl::expected<ParallelTensorShape, std::string>
    get_output_shape(ReductionAttrs const &attrs,
                     ParallelTensorShape const &input_shape);

ParallelTensorDimDegrees get_output_parallel_dim_degrees(
    ReductionAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees);

OperatorTaskSpace
    get_operator_task_space(ReductionAttrs const &attrs,
                            ParallelTensorDimDegrees const &input_degrees);

} // namespace FlexFlow

#endif
