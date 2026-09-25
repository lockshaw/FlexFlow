#include "utils/containers/binary_cartesian_product.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using A = ordered_value_type<0>;
using B = ordered_value_type<1>;

template
binary_cartesian_product_container<
  typename std::set<A>::const_iterator,
  typename std::set<B>::const_iterator
>
    binary_cartesian_product(std::set<A> const &, std::set<B> const &);

} // namespace FlexFlow
