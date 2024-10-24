#include "utils/many_to_one/many_to_one.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using L = value_type<0>;
using R = value_type<1>;

template struct ManyToOne<L, R>;

} // namespace FlexFlow
