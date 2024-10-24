#include "utils/one_to_many/one_to_many.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using L = value_type<0>;
using R = value_type<1>;

template struct OneToMany<L, R>;

} // namespace FlexFlow

