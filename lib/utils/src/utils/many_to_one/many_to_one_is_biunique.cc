#include "utils/many_to_one/many_to_one_is_biunique.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using L = ordered_value_type<0>;
using R = ordered_value_type<1>;

template bool many_to_one_is_biunique(ManyToOne<L, R> const &);

} // namespace FlexFlow
