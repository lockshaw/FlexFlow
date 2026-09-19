#include "utils/containers/multiset_intersection.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template
  std::multiset<T> multiset_intersection(std::multiset<T> const &,
                                         std::multiset<T> const &);

} // namespace FlexFlow
