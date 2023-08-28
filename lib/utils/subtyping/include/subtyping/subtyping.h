#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_SUBTYPING_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_SUBTYPING_H

#include "utils/type_traits.h"
#include "utils/type_traits_core.h"
#include "boost/preprocessor/repetition/enum_params.hpp"
#include "boost/preprocessor/selection/max.hpp"
#include "boost/preprocessor/control/if.hpp"
#include "boost/preprocessor/comparison/equal.hpp"
#include "boost/preprocessor/comparison/greater.hpp"
#include <boost/preprocessor/control/expr_if.hpp>
#include <boost/preprocessor/punctuation/remove_parens.hpp>
#include <boost/preprocessor/facilities/intercept.hpp>
#include <boost/preprocessor/facilities/overload.hpp>

namespace FlexFlow {

template <typename T>
struct tag_parent_type;

template <typename T>
using tag_parent_type_t = typename tag_parent_type<T>::type;

template <typename T>
struct tag_has_parent_type;

template <typename C, typename P, typename Enable = void>
struct tag_is_subtype_of : std::false_type {};

template <typename C, typename P>
struct tag_is_subtype_of<C, P, enable_if_t<std::is_same<C, P>::value>> : std::true_type {};

template <typename C, typename P>
struct tag_is_subtype_of<C, P, enable_if_t<tag_has_parent_type<C>::value && !std::is_same<C, P>::value>>
  : tag_is_subtype_of<tag_parent_type_t<C>, P> {};

template <typename T>
struct tag_type_to_impl_type;

template <typename T>
using tag_type_to_impl_type_t = typename tag_type_to_impl_type<T>::type;

template <typename T>
struct impl_type_to_tag_type;

template <typename T>
using impl_type_to_tag_type_t = typename impl_type_to_tag_type<T>::type;

template <typename T>
struct is_tag_type : std::false_type {};

template <typename T>
struct is_impl_type : std::false_type {};

template <template <typename> class T>
struct is_subtyping_forest : std::false_type {};

#define TEMPLATE_DECL(N_ARGS) \
  BOOST_PP_REMOVE_PARENS(BOOST_PP_EXPR_IF(N_ARGS, (template < BOOST_PP_ENUM_PARAMS(N_ARGS, typename T) >)))

#define TEMPLATE_SPECIALIZE(N_ARGS) \
  template < BOOST_PP_ENUM_PARAMS(N_ARGS, typename T) >

#define TEMPLATE_USE(NAME, N_ARGS) \
  BOOST_PP_REMOVE_PARENS(BOOST_PP_IF(BOOST_PP_GREATER(N_ARGS, 0), (NAME <BOOST_PP_ENUM_PARAMS(N_ARGS, T)>), NAME))

#define TEMPLATE_CONCRETIZE(NAME, N_ARGS) \
  BOOST_PP_REMOVE_PARENS(BOOST_PP_IF(BOOST_PP_GREATER(N_ARGS, 0), (NAME <BOOST_PP_ENUM_PARAMS(N_ARGS, int BOOST_PP_INTERCEPT )>), NAME))

#define CHECK_NOT_TAG_TYPE(TYPENAME, N_TEMPLATE_PARAMS) \
  static_assert(!is_tag_type< TEMPLATE_CONCRETIZE(TYPENAME, N_TEMPLATE_PARAMS) >::value, #TYPENAME " should not be a tag type");

#define CHECK_NOT_IMPL_TYPE(TYPENAME, N_TEMPLATE_PARAMS) \
  static_assert(!is_impl_type< TEMPLATE_CONCRETIZE(TYPENAME, N_TEMPLATE_PARAMS) >::value, #TYPENAME " should not be an impl type");

#define CHECK_NOT_IS_SUBTYPING_FOREST(TYPENAME) \
  static_assert(!is_subtyping_forest<TYPENAME>::value, #TYPENAME " should not be a subtyping forest");

#define CHECK_IS_TAG_TYPE(TYPENAME, N_TEMPLATE_PARAMS) \
  static_assert(is_tag_type< TEMPLATE_CONCRETIZE(TYPENAME, N_TEMPLATE_PARAMS) >::value, #TYPENAME " should be a tag type");

#define CHECK_IS_IMPL_TYPE(TYPENAME, N_TEMPLATE_PARAMS) \
  static_assert(is_impl_type< TEMPLATE_CONCRETIZE(TYPENAME, N_TEMPLATE_PARAMS) >::value, #TYPENAME " should be an impl type");

#define CHECK_IS_SUBTYPING_FOREST(TYPENAME) \
  static_assert(is_subtyping_forest<TYPENAME>::value, #TYPENAME " should be a subtyping forest");

#define MAKE_SUBTYPING_FOREST(TAG_NAME) \
  template <typename T> struct TAG_NAME; \
  template <> struct is_subtyping_forest<TAG_NAME> : std::true_type {};

#define MAKE_TAG_TYPE(TAG_NAME, IMPL_TYPE, N_IMPL_TEMPLATE_ARGS) \
  TEMPLATE_SPECIALIZE(N_IMPL_TEMPLATE_ARGS) \
  struct is_tag_type<TAG_NAME< TEMPLATE_USE(IMPL_TYPE, N_IMPL_TEMPLATE_ARGS) >> : std::true_type {}; \
  TEMPLATE_SPECIALIZE(N_IMPL_TEMPLATE_ARGS) \
  struct is_impl_type< TEMPLATE_USE(IMPL_TYPE, N_IMPL_TEMPLATE_ARGS) > : std::true_type {}; \
  TEMPLATE_SPECIALIZE(N_IMPL_TEMPLATE_ARGS) \
  struct tag_type_to_impl_type<TAG_NAME< TEMPLATE_USE(IMPL_TYPE, N_IMPL_TEMPLATE_ARGS) >> \
    : type_identity< TEMPLATE_USE(IMPL_TYPE, N_IMPL_TEMPLATE_ARGS) > {}; \
  TEMPLATE_SPECIALIZE(N_IMPL_TEMPLATE_ARGS) \
  struct impl_type_to_tag_type< TEMPLATE_USE(IMPL_TYPE, N_IMPL_TEMPLATE_ARGS) > \
    : type_identity<TAG_NAME< TEMPLATE_USE(IMPL_TYPE, N_IMPL_TEMPLATE_ARGS) >> {}; 

#define MAKE_SUBTYPING_FOREST_ROOT_1(FOREST_NAME, IMPL_TYPE) \
  MAKE_SUBTYPING_FOREST_ROOT_2(FOREST_NAME, IMPL_TYPE, 0)

#define MAKE_SUBTYPING_FOREST_ROOT_2(FOREST_NAME, IMPL_TYPE, N_IMPL_TEMPLATE_ARGS) \
  CHECK_IS_SUBTYPING_FOREST(FOREST_NAME); \
  TEMPLATE_SPECIALIZE(N_IMPL_TEMPLATE_ARGS) \
  struct FOREST_NAME< TEMPLATE_USE(IMPL_TYPE, N_IMPL_TEMPLATE_ARGS) > {}; \
  MAKE_TAG_TYPE(FOREST_NAME, IMPL_TYPE, N_IMPL_TEMPLATE_ARGS); \
  TEMPLATE_SPECIALIZE(N_IMPL_TEMPLATE_ARGS) \
  struct tag_has_parent_type<FOREST_NAME< TEMPLATE_USE(IMPL_TYPE, N_IMPL_TEMPLATE_ARGS) >> : std::false_type {};

#define MAKE_SUBTYPING_FOREST_ROOT(FOREST_NAME, ...) BOOST_PP_OVERLOAD(MAKE_SUBTYPING_FOREST_ROOT_, __VA_ARGS__)(FOREST_NAME, __VA_ARGS__)

#define MAKE_SUBTYPING_RELATION_1(FOREST_NAME, IMPL_TYPE, PARENT) \
  MAKE_SUBTYPING_RELATION_3(FOREST_NAME, IMPL_TYPE, 0, PARENT, 0)

#define MAKE_SUBTYPING_RELATION_2(FOREST_NAME, IMPL_TYPE, N_IMPL_TEMPLATE_ARGS, PARENT) \
  MAKE_SUBTYPING_RELATION_3(FOREST_NAME, IMPL_TYPE, N_IMPL_TEMPLATE_ARGS, PARENT, 0)

#define MAKE_SUBTYPING_RELATION_3(FOREST_NAME, IMPL_TYPE, N_IMPL_TEMPLATE_ARGS, PARENT, N_PARENT_TEMPLATE_ARGS) \
  CHECK_IS_SUBTYPING_FOREST(FOREST_NAME); \
  static_assert(N_IMPL_TEMPLATE_ARGS >= N_PARENT_TEMPLATE_ARGS, "Number of child template parameters " #N_IMPL_TEMPLATE_ARGS " must be >= the number of parent template parameters " #N_PARENT_TEMPLATE_ARGS); \
  CHECK_IS_IMPL_TYPE(PARENT, N_PARENT_TEMPLATE_ARGS); \
  TEMPLATE_SPECIALIZE(N_IMPL_TEMPLATE_ARGS) \
  struct FOREST_NAME< TEMPLATE_USE(IMPL_TYPE, N_IMPL_TEMPLATE_ARGS) > \
    : public impl_type_to_tag_type_t < TEMPLATE_USE(PARENT, N_PARENT_TEMPLATE_ARGS) > {}; \
  MAKE_TAG_TYPE(FOREST_NAME, IMPL_TYPE, N_IMPL_TEMPLATE_ARGS); \
  TEMPLATE_SPECIALIZE(N_IMPL_TEMPLATE_ARGS) \
  struct tag_parent_type <FOREST_NAME< TEMPLATE_USE(IMPL_TYPE, N_IMPL_TEMPLATE_ARGS) >> \
    : impl_type_to_tag_type< TEMPLATE_USE(PARENT, N_PARENT_TEMPLATE_ARGS) > {}; \
  TEMPLATE_SPECIALIZE(N_IMPL_TEMPLATE_ARGS) \
  struct tag_has_parent_type<FOREST_NAME< TEMPLATE_USE(IMPL_TYPE, N_IMPL_TEMPLATE_ARGS) >> : std::true_type {}; \
  static_assert(std::is_convertible<TEMPLATE_CONCRETIZE(IMPL_TYPE, N_IMPL_TEMPLATE_ARGS), TEMPLATE_CONCRETIZE(PARENT, N_PARENT_TEMPLATE_ARGS)>::value, "To create a subtyping relation from child " #IMPL_TYPE " to parent " #PARENT ", " #IMPL_TYPE " must be convertible to " #PARENT);

#define MAKE_SUBTYPING_RELATION(FOREST_NAME, IMPL_TYPE, ...) BOOST_PP_OVERLOAD(MAKE_SUBTYPING_RELATION_, __VA_ARGS__)(FOREST_NAME, IMPL_TYPE, __VA_ARGS__)

  /* static_assert(std::is_same<tag_parent_type_t<TAG_NAME<IMPL_TYPE<BOOST_PP_ENUM_PARAMS(N_IMPL_TEMPLATE_ARGS, T)>>>, impl_type_to_tag_type_t<PARENT>>::value, ""); \ */

template <typename RequestedTag, typename CurrentImpl>
enable_if_t<
  std::is_same<RequestedTag, tag_parent_type_t<impl_type_to_tag_type_t<CurrentImpl>>>::value,
  tag_type_to_impl_type_t<RequestedTag>>
coerce(CurrentImpl const &t) {
  /* CHECK_IS_TAG_TYPE(RequestedTag); */
  /* CHECK_IS_IMPL_TYPE(CurrentImpl); */

  return static_cast<tag_type_to_impl_type_t<RequestedTag>>(t);
}

template <typename RequestedTag, typename CurrentImpl>
enable_if_t<
  !std::is_same<RequestedTag, tag_parent_type_t<impl_type_to_tag_type_t<CurrentImpl>>>::value,
  tag_type_to_impl_type_t<RequestedTag>> 
coerce(CurrentImpl const &t) {
  /* CHECK_IS_TAG_TYPE(RequestedTag); */
  /* CHECK_IS_IMPL_TYPE(CurrentImpl); */
  static_assert(tag_is_subtype_of<impl_type_to_tag_type_t<CurrentImpl>, RequestedTag>::value, "Requested coercion violates subtyping structure");

  using CurrentParentTag = tag_parent_type_t<impl_type_to_tag_type_t<CurrentImpl>>;
  return coerce<RequestedTag>(coerce<CurrentParentTag>(t));
}

template <typename RequestedTag, typename CurrentImpl>
tag_type_to_impl_type_t<RequestedTag>
coerce(CurrentImpl const &ci, RequestedTag const &) {
  return coerce<RequestedTag>(ci);
}

template <typename Impl>
impl_type_to_tag_type_t<Impl> create_tag(Impl const &) {
  return impl_type_to_tag_type_t<Impl>{};
}

}

#endif
