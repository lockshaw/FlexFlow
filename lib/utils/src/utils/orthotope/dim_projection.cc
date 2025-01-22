#include "utils/orthotope/dim_projection.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using L = value_type<0>;
using R = value_type<1>;

template
  std::unordered_set<L> input_dims_of_projection(DimProjection<L, R> const &);

template
  std::unordered_set<R> output_dims_of_projection(DimProjection<L, R> const &);

} // namespace FlexFlow
