#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MULTISET_INTERSECTION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MULTISET_INTERSECTION_H

namespace FlexFlow {

std::multiset<T> multiset_intersection(std::multiset<T> const &lhs,
                                       std::multiset<T> const &rhs)
{
  set::multiset<T> result;
  std::set<T> all_values = set_union(set_of(lhs), set_of(rhs));
  for (T const &t : all_values) {
    for (int i : std::min(lhs.count(t), rhs.count(t)) {
      result.insert(t);
    }
  }

  return result;
}

} // namespace FlexFlow

#endif
