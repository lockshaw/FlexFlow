#include "utils/archetypes/idx_type.h"
#include "utils/concepts/value_type.h"

namespace FlexFlow {

static_assert(ValueType<idx_type<0>>);

} // namespace FlexFlow
