#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_CAST_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_CAST_H

#include "op-attrs/ops/cast_attrs.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/shard_signature_instance.h"
#include "op-attrs/standard_operator_task_group.h"
#include "op-attrs/tensor_shape.dtg.h"

namespace FlexFlow {

TensorShape cast_get_output_shape(CastAttrs const &, TensorShape const &);

ParallelTensorDimDegrees
    cast_get_output_parallel_dim_degrees(CastAttrs const &,
                                         ParallelTensorDimDegrees const &);

ParallelTensorShape cast_get_output_parallel_shape(CastAttrs const &,
                                                   ParallelTensorShape const &);

StandardOperatorTaskGroup
    cast_get_task_group(CastAttrs const &attrs,
                        ParallelTensorDimDegrees const &input_degrees);

} // namespace FlexFlow

#endif
