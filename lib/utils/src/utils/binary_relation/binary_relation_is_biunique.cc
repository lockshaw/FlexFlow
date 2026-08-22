#include "utils/binary_relation/binary_relation_is_biunique.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using L = ordered_value_type<0>;
using R = ordered_value_type<1>;

template bool binary_relation_is_biunique(BinaryRelation<L, R> const &);

} // namespace FlexFlow
