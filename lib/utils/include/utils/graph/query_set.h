#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_QUERY_SET_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_GRAPH_QUERY_SET_H

#include "utils/bidict/bidict.h"
#include "utils/containers/contains.h"
#include "utils/containers/filter.h"
#include "utils/containers/filter_keys.h"
#include "utils/containers/intersection.h"
#include "utils/containers/set_union.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/exception.h"
#include "utils/fmt/unordered_set.h"
#include "utils/hash-utils.h"
#include "utils/hash/set.h"
#include "utils/optional.h"
#include <optional>
#include <set>
#include <unordered_set>
#include "utils/hash/unordered_set.h"

namespace FlexFlow {

template <typename T>
struct query_set {
  query_set() = delete;
  query_set(T const &t) : query(std::unordered_set<T>{t}) {}

  query_set(std::unordered_set<T> const &query)
      : query(std::unordered_set<T>{query.cbegin(), query.cend()}) {}

  query_set(std::optional<std::unordered_set<T>> const &query)
      : query(transform(query, [](std::unordered_set<T> const &s) {
          return std::unordered_set<T>{s.cbegin(), s.cend()};
        })) {}

  query_set(std::initializer_list<T> const &l)
      : query_set(std::unordered_set<T>{l}) {}

  friend bool operator==(query_set const &lhs, query_set const &rhs) {
    return lhs.query == rhs.query;
  }

  friend bool operator!=(query_set const &lhs, query_set const &rhs) {
    return lhs.query != rhs.query;
  }

  // friend bool operator<(query_set const &lhs, query_set const &rhs) {
  //   return lhs.query < rhs.query;
  // }
  //
  friend bool is_matchall(query_set const &q) {
    return !q.query.has_value();
  }

  bool includes(T const &t) const {
    return !query.has_value() || contains(query.value(), t);
  }

  std::unordered_set<T> apply(std::unordered_set<T> input) {
    if (!this->query.has_value()) {
      return input;
    }

    return intersection(input, this->query.value());
  }

  friend std::unordered_set<T> allowed_values(query_set const &q) {
    assert(!is_matchall(q));
    std::unordered_set<T> query_value = q.query.value();
    return std::unordered_set<T>{query_value.begin(), query_value.end()};
  }

  static query_set<T> matchall() {
    return {std::nullopt};
  }

  static query_set<T> match_none() {
    return {std::unordered_set<T>{}};
  }

  std::optional<std::unordered_set<T>> const &value() const {
    return this->query;
  }

private:
  std::optional<std::unordered_set<T>> query;
};

template <typename T>
std::string format_as(query_set<T> const &q) {
  if (is_matchall(q)) {
    return "(all)";
  } else {
    return fmt::format(FMT_STRING("query_set({})"), allowed_values(q));
  }
}

template <typename T>
struct delegate_ostream_operator<query_set<T>> : std::true_type {};

template <typename T>
query_set<T> matchall() {
  return query_set<T>::matchall();
}

template <typename T>
bool includes(query_set<T> const &q, T const &v) {
  return q.includes(v);
}

template <typename T, typename C>
std::unordered_set<T> apply_query(query_set<T> const &q, C const &c) {
  std::unordered_set<T> c_set = unordered_set_of(c);
  if (is_matchall(q)) {
    return c_set;
  }

  return q.apply(c_set);
}

template <typename C,
          typename K = typename C::key_type,
          typename V = typename C::mapped_type>
std::unordered_map<K, V> query_keys(query_set<K> const &q, C const &m) {
  if (is_matchall(q)) {
    return m;
  }
  return filter_keys(m, [&](K const &key) { return includes(q, key); });
}

template <typename C,
          typename K = typename C::key_type,
          typename V = typename C::mapped_type>
std::unordered_map<K, V> query_values(query_set<V> const &q, C const &m) {
  if (is_matchall(q)) {
    return m;
  }
  return filter_values(m, [&](V const &value) { return includes(q, value); });
}

template <typename T>
query_set<T> query_intersection(query_set<T> const &lhs,
                                query_set<T> const &rhs) {
  if (is_matchall(lhs)) {
    return rhs;
  } else if (is_matchall(rhs)) {
    return lhs;
  } else {
    return intersection(allowed_values(lhs), allowed_values(rhs));
  }
}

template <typename T>
query_set<T> query_union(query_set<T> const &lhs, query_set<T> const &rhs) {
  if (is_matchall(lhs) || is_matchall(rhs)) {
    return query_set<T>::matchall();
  } else {
    return set_union(allowed_values(lhs), allowed_values(rhs));
  }
}

} // namespace FlexFlow

namespace std {

template <typename T>
struct hash<::FlexFlow::query_set<T>> {
  size_t operator()(::FlexFlow::query_set<T> const &q) const {
    return ::FlexFlow::get_std_hash(q.value());
  }
};

} // namespace std

#endif
