// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/op-attrs/include/op-attrs/ops/transpose_attrs.struct.toml
/* proj-data
{
  "generated_from": "87f6e4db4b66d564530994773c0ecef4"
}
*/

#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_TRANSPOSE_ATTRS_DTG_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_TRANSPOSE_ATTRS_DTG_H

#include "fmt/format.h"
#include "nlohmann/json.hpp"
#include "op-attrs/ff_dim.dtg.h"
#include "op-attrs/ff_dim.h"
#include "rapidcheck.h"
#include "utils/stack_vector.h"
#include <functional>
#include <ostream>
#include <tuple>

namespace FlexFlow {
struct TransposeAttrs {
  TransposeAttrs() = delete;
  TransposeAttrs(::FlexFlow::stack_vector<::FlexFlow::ff_dim_t,
                                          MAX_TENSOR_DIM> const &perm);

  bool operator==(TransposeAttrs const &) const;
  bool operator!=(TransposeAttrs const &) const;
  bool operator<(TransposeAttrs const &) const;
  bool operator>(TransposeAttrs const &) const;
  bool operator<=(TransposeAttrs const &) const;
  bool operator>=(TransposeAttrs const &) const;
  ::FlexFlow::stack_vector<::FlexFlow::ff_dim_t, MAX_TENSOR_DIM> perm;
};
} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::TransposeAttrs> {
  size_t operator()(FlexFlow::TransposeAttrs const &) const;
};
} // namespace std

namespace nlohmann {
template <>
struct adl_serializer<FlexFlow::TransposeAttrs> {
  static FlexFlow::TransposeAttrs from_json(json const &);
  static void to_json(json &, FlexFlow::TransposeAttrs const &);
};
} // namespace nlohmann

namespace rc {
template <>
struct Arbitrary<FlexFlow::TransposeAttrs> {
  static Gen<FlexFlow::TransposeAttrs> arbitrary();
};
} // namespace rc

namespace FlexFlow {
std::string format_as(TransposeAttrs const &);
std::ostream &operator<<(std::ostream &, TransposeAttrs const &);
} // namespace FlexFlow

#endif // _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_TRANSPOSE_ATTRS_DTG_H
