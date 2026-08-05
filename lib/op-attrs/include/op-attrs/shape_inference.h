#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_SHAPE_INFERENCE_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_SHAPE_INFERENCE_H

#include "op-attrs/computation_graph_op_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/pcg_operator_attrs.dtg.h"
#include "op-attrs/tensor_slot_name.dtg.h"
#include <vector>

namespace FlexFlow {

std::map<TensorSlotName, TensorShape> get_output_shapes(
    ComputationGraphOpAttrs const &,
    std::map<TensorSlotName, TensorShape> const &input_shapes);

std::map<TensorSlotName, TensorShape> get_weight_shapes(
    ComputationGraphOpAttrs const &,
    std::map<TensorSlotName, TensorShape> const &input_shapes);

std::map<TensorSlotName, ParallelTensorShape> get_output_shapes(
    PCGOperatorAttrs const &,
    std::map<TensorSlotName, ParallelTensorShape> const &input_shapes);

std::map<TensorSlotName, ParallelTensorShape> get_weight_shapes(
    PCGOperatorAttrs const &,
    std::map<TensorSlotName, ParallelTensorShape> const &input_shapes);

std::map<TensorSlotName, ParallelTensorDimDegrees>
    infer_output_degrees(
        PCGOperatorAttrs const &,
        std::map<TensorSlotName, ParallelTensorDimDegrees> const
            &input_degrees);

std::map<TensorSlotName, ParallelTensorDimDegrees>
    infer_weight_degrees(
        PCGOperatorAttrs const &,
        std::map<TensorSlotName, ParallelTensorDimDegrees> const
            &input_degrees);

} // namespace FlexFlow

#endif
