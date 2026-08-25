#include "utils/binary_relation/binary_relation_is_right_k_unique.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using L = ordered_value_type<0>;
using R = ordered_value_type<1>;

template
  std::optional<nonnegative_int> binary_relation_is_right_k_unique(BinaryRelation<L, R> const &);

} // namespace FlexFlow
