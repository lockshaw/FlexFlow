#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_SEQUENCE_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_SEQUENCE_H

#include "utils/tuple.h"
#include <optional>
#include <utility>

namespace FlexFlow {

template <int... S>
struct seq {};

template <typename Seq>
struct seq_head;

template <int X, int... S>
struct seq_head<seq<X, S...>> : std::integral_constant<int, X> {};

template <>
struct seq_head<seq<>> : std::integral_constant<int, -1> {};

template <typename Seq>
struct seq_tail;

template <int X, int... S>
struct seq_tail<seq<X, S...>> {
  using type = seq<S...>;
};

template <>
struct seq_tail<seq<>> {
  using type = seq<>;
};

template <int X, int... S>
struct seq_prepend {
  using type = seq<X, S...>;
};

template <typename Rest, int Head>
struct seq_append;

template <int X, int... S>
struct seq_append<seq<S...>, X> {
  using type = seq<S..., X>;
};

template <int n>
struct seq_count {
  using type = typename seq_append<typename seq_count<(n - 1)>::type, n>::type;
};

template <>
struct seq_count<-1> {
  using type = seq<>;
};

template <int n>
using seq_count_t = typename seq_count<n>::type;

template <typename... Args>
struct seq_enumerate_args {
  using type = seq_count_t<(int)(sizeof...(Args)) - 1>;
};

template <typename... Args>
using seq_enumerate_args_t = typename seq_enumerate_args<Args...>::type;

template <typename T>
struct seq_enumerate;

template <typename... Args>
struct seq_enumerate<std::tuple<Args...>> : seq_enumerate_args<Args...> {};

template <typename T>
using seq_enumerate_t = typename seq_enumerate<T>::type;

} // namespace FlexFlow

#endif
