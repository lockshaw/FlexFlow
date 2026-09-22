#include "utils/containers/multiset_intersection.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using T = ordered_value_type<0>;

template
  std::multiset<T> multiset_intersection(std::multiset<T> const &,
                                         std::multiset<T> const &);

template
  std::optional<std::multiset<T>> 
    multiset_intersection(std::vector<std::multiset<T>> const &);

template
  std::multiset<T> multiset_intersection1(std::vector<std::multiset<T>> const &);

} // namespace FlexFlow
