#include "op-attrs/ops/gather.h"
#include "utils/not_implemented.h"

namespace FlexFlow {

TensorShape gather_get_output_shape(GatherAttrs const &,
                                    TensorShape const &input,
                                    TensorShape const &index) {
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

ParallelTensorDimDegrees gather_get_output_parallel_dim_degrees(
    GatherAttrs const &,
    ParallelTensorDimDegrees const &input,
    ParallelTensorDimDegrees const &index) {
  NOT_IMPLEMENTED();
}

ParallelTensorShape
    gather_get_output_parallel_shape(GatherAttrs const &,
                                     ParallelTensorShape const &input,
                                     ParallelTensorShape const &index) {
  NOT_IMPLEMENTED();
}

OperatorTaskSpace gather_get_operator_task_space(
    GatherAttrs const &attrs,
    ParallelTensorDimDegrees const &input_degrees,
    ParallelTensorDimDegrees const &index_degrees) {
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping
    gather_get_operator_to_input_mapping(
        GatherAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees,
        ParallelTensorDimDegrees const &index_degrees) {
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping
    gather_get_operator_to_index_mapping(
        GatherAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees,
        ParallelTensorDimDegrees const &index_degrees) {
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping
    gather_get_operator_to_output_mapping(
        GatherAttrs const &attrs,
        ParallelTensorDimDegrees const &input_degrees,
        ParallelTensorDimDegrees const &index_degrees) {
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
