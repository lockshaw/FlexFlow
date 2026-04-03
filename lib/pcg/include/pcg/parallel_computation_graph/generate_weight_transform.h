#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_PARALLEL_COMPUTATION_GRAPH_GENERATE_WEIGHT_TRANSFORM_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_PARALLEL_COMPUTATION_GRAPH_GENERATE_WEIGHT_TRANSFORM_H

#include "op-attrs/parallel_op_attrs.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"

namespace FlexFlow {

std::unordered_set<ParallelOpAttrs>
    generate_weight_transform(ParallelTensorDimDegrees const &goal);

} // namespace FlexFlow

#endif
