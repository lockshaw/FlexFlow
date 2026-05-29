#include "utils/one_to_many/one_to_many_is_biunique.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using L = value_type<0>;
using R = value_type<1>;

template bool one_to_many_is_biunique(OneToMany<L, R> const &);

} // namespace FlexFlow
