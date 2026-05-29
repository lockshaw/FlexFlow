#include "utils/relation/hemiunique_binary_relation.h"
#include "utils/archetypes/value_type.h"

using L = ::FlexFlow::value_type<0>;
using R = ::FlexFlow::value_type<1>;

namespace FlexFlow {

template struct HemiuniqueBinaryRelation<L, R>;

template std::string format_as(HemiuniqueBinaryRelation<L, R> const &);

template std::ostream &operator<<(std::ostream &, HemiuniqueBinaryRelation<L, R> const &);

} // namespace FlexFlow

namespace std {

template struct hash<::FlexFlow::HemiuniqueBinaryRelation<L, R>>;

}
