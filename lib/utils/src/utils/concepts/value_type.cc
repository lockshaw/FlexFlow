#include "utils/concepts/value_type.h"
#include <string>

namespace FlexFlow {

static_assert(ValueType<int>);
static_assert(ValueType<std::string>);

} // namespace FlexFlow
