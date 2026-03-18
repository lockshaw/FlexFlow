#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_DATA_PARALLELISM_DATA_PARALLELISM_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_DATA_PARALLELISM_DATA_PARALLELISM_H

#include "compiler/search_result.dtg.h"
#include "pcg/computation_graph.dtg.h"

namespace FlexFlow {

SearchResult
  apply_data_parallelism(ComputationGraph const &pcg,
                         int_ge_two degree);


} // namespace FlexFlow

#endif
