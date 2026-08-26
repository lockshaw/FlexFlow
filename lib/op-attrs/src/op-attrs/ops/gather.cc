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
  ParallelTensorDimDegrees const &index)
{
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

ParallelTensorShape gather_get_output_parallel_shape(GatherAttrs const &,
                                     ParallelTensorShape const &input,
                                     ParallelTensorShape const &index) {
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

OperatorTaskSpace gather_get_operator_task_space(
    GatherAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping gather_get_operator_to_input_mapping(
    GatherAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping gather_get_operator_to_index_mapping(
    GatherAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}

OperatorSpaceToParallelTensorSpaceBiuniqueMapping gather_get_operator_to_output_mapping(
    GatherAttrs const &attrs, ParallelTensorDimDegrees const &input_degrees)
{
  // TODO(@lockshaw)(#pr):
  NOT_IMPLEMENTED();
}


/* bool GatherAttrs::is_valid(ParallelTensorShape const &lhs,
 * ParallelTensorShape const &rhs) const { */
/*   if (lhs.num_dims() != rhs.num_dims()) { */
/*     return false; */
/*   } */
/*   for (int i = 0; i < lhs.num_dims(); i++) { */
/*     if (i != this->legion_dim && */
/*         lhs.at(i).size < rhs.at(i).size) { */
/*       return false; */
/*     } */
/*   } */
/*   return true; */
/* } */

} // namespace FlexFlow
