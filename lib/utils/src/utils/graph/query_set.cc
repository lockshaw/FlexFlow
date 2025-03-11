#include "utils/graph/query_set.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template struct query_set<T>;

}
