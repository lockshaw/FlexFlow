#include "utils/nonnegative_int/num_elements.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template
  nonnegative_int num_elements(std::vector<T> const &);

template
  nonnegative_int num_elements(std::list<T> const &);

template
  nonnegative_int num_elements(std::set<T> const &);

template
  nonnegative_int num_elements(std::unordered_set<T> const &);

} // namespace FlexFlow
