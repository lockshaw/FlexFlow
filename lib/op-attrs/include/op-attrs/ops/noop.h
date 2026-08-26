#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_NOOP_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_NOOP_H

#include "op-attrs/ops/noop_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/operator_task_space.dtg.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_biunique_mapping.dtg.h"

namespace FlexFlow {

TensorShape noop_get_output_shape(NoopAttrs const &, TensorShape const &);
ParallelTensorDimDegrees noop_get_output_parallel_dim_degrees(NoopAttrs const &,
                                                              ParallelTensorDimDegrees const &);
ParallelTensorShape noop_get_output_parallel_shape(NoopAttrs const &,
                                                   ParallelTensorShape const &);

OperatorTaskSpace noop_get_operator_task_space(
    NoopAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);

OperatorSpaceToParallelTensorSpaceBiuniqueMapping noop_get_operator_to_input_mapping(
    NoopAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);

OperatorSpaceToParallelTensorSpaceBiuniqueMapping noop_get_operator_to_output_mapping(
    NoopAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);

} // namespace FlexFlow

#endif
