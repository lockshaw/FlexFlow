#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_CAST_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_CAST_H

#include "op-attrs/ops/cast_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "utils/relation/compose_hemiunique_binary_relations.h"
#include "op-attrs/operator_task_space.dtg.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_biunique_mapping.dtg.h"

namespace FlexFlow {

TensorShape cast_get_output_shape(CastAttrs const &, TensorShape const &);

ParallelTensorDimDegrees cast_get_output_parallel_dim_degrees(
  CastAttrs const &, ParallelTensorDimDegrees const &);

ParallelTensorShape
    cast_get_output_parallel_shape(CastAttrs const &, ParallelTensorShape const &);

OperatorTaskSpace cast_get_operator_task_space(
    CastAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);

OperatorSpaceToParallelTensorSpaceBiuniqueMapping cast_get_operator_to_input_mapping(
    CastAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);

OperatorSpaceToParallelTensorSpaceBiuniqueMapping cast_get_operator_to_output_mapping(
    CastAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees);


} // namespace FlexFlow

#endif
