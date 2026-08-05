#include "utils/one_to_many/one_to_many_is_biunique.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using L = ordered_value_type<0>;
using R = ordered_value_type<1>;

template bool one_to_many_is_biunique(OneToMany<L, R> const &);

} // namespace FlexFlow
