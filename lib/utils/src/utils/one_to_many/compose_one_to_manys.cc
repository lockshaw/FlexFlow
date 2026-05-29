#include "utils/one_to_many/compose_one_to_manys.h"

namespace FlexFlow {

using T1 = value_type<0>;
using T2 = value_type<1>;
using T3 = value_type<2>;

template
  OneToMany<T1, T3> compose_one_to_manys(OneToMany<T1, T2> const &,
                                         OneToMany<T2, T3> const &);

} // namespace FlexFlow
