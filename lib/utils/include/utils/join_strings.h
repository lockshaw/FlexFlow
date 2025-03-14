#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_JOIN_STRINGS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_JOIN_STRINGS_H

#include <sstream>
#include <string>

namespace FlexFlow {

template <typename InputIt, typename F>
std::string join_strings(InputIt first,
                         InputIt last,
                         std::string const &delimiter,
                         F const &f) {
  std::ostringstream oss;
  bool first_iter = true;
  for (; first != last; first++) {
    if (!first_iter) {
      oss << delimiter;
    }
    oss << f(*first);
    first_iter = false;
  }
  return oss.str();
}

template <typename InputIt>
std::string
    join_strings(InputIt first, InputIt last, std::string const &delimiter) {
  using Ref = typename InputIt::reference;
  return join_strings<InputIt>(first, last, delimiter, [](Ref r) { return r; });
}

template <typename Container>
std::string join_strings(Container const &c, std::string const &delimiter) {
  return join_strings(c.cbegin(), c.cend(), delimiter);
}

template <typename Container, typename F>
std::string
    join_strings(Container const &c, std::string const &delimiter, F const &f) {
  return join_strings(c.cbegin(), c.cend(), delimiter, f);
}

} // namespace FlexFlow

#endif
