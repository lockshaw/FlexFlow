#include "utils/containers/take_until.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template
  std::vector<T> take_until(std::vector<T> const &,
                            std::function<bool(T const &)> const &);

} // namespace FlexFlow
