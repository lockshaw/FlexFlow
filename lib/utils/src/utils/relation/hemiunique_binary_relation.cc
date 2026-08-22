#include "utils/relation/hemiunique_binary_relation.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/jsonable_ordered_value_type.h"

namespace FlexFlow {

using L = ordered_value_type<0>;
using R = ordered_value_type<1>;

template struct HemiuniqueBinaryRelation<L, R>;

template std::string format_as(HemiuniqueBinaryRelation<L, R> const &);

template std::ostream &operator<<(std::ostream &,
                                  HemiuniqueBinaryRelation<L, R> const &);

} // namespace FlexFlow

namespace nlohmann {

using L = ::FlexFlow::jsonable_ordered_value_type<0>;
using R = ::FlexFlow::jsonable_ordered_value_type<1>;

template struct adl_serializer<::FlexFlow::HemiuniqueBinaryRelation<L, R>>;

}

namespace std {

using L = ::FlexFlow::ordered_value_type<0>;
using R = ::FlexFlow::ordered_value_type<1>;

template struct hash<::FlexFlow::HemiuniqueBinaryRelation<L, R>>;

}
