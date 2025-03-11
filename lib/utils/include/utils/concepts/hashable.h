#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONCEPTS_HASHABLE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONCEPTS_HASHABLE_H

namespace FlexFlow {

// from cppreference: https://en.cppreference.com/w/cpp/language/constraints
template<typename T>
concept Hashable = requires(T a) {
    { std::hash<T>{}(a) } -> std::convertible_to<std::size_t>;
};

} // namespace FlexFlow

#endif
