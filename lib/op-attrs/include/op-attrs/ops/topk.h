#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_TOPK_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_TOPK_H

#include "op-attrs/operator_task_space.dtg.h"
#include "op-attrs/ops/topk_attrs.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/standard_operator_task_group.h"
#include "op-attrs/tensor_shape.dtg.h"

namespace FlexFlow {

TensorShape topk_get_output_shape(TopKAttrs const &, TensorShape const &);

ParallelTensorDimDegrees
    topk_get_output_parallel_dim_degrees(TopKAttrs const &,
                                         ParallelTensorDimDegrees const &);

ParallelTensorShape
    topk_get_output_parallel_shape(TopKAttrs const &attrs,
                                   ParallelTensorShape const &input_shape);

StandardOperatorTaskGroup topk_get_task_group(TopKAttrs const &,
                                              ParallelTensorDimDegrees const &);

} // namespace FlexFlow

#endif
