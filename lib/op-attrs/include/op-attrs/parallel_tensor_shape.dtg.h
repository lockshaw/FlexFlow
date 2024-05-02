// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/op-attrs/include/op-attrs/parallel_tensor_shape.struct.toml
/* proj-data
{
  "generated_from": "b2d36c9212916e66569af4e958c893f4"
}
*/

#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_TENSOR_SHAPE_DTG_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_TENSOR_SHAPE_DTG_H

#include "fmt/format.h"
#include "nlohmann/json.hpp"
#include "op-attrs/datatype.h"
#include "op-attrs/parallel_tensor_dims.h"
#include <functional>
#include <ostream>
#include <tuple>

namespace FlexFlow {
struct ParallelTensorShape {
  ParallelTensorShape() = delete;
  ParallelTensorShape(::FlexFlow::ParallelTensorDims const &dims,
                      ::FlexFlow::DataType const &data_type);

  bool operator==(ParallelTensorShape const &) const;
  bool operator!=(ParallelTensorShape const &) const;
  bool operator<(ParallelTensorShape const &) const;
  bool operator>(ParallelTensorShape const &) const;
  bool operator<=(ParallelTensorShape const &) const;
  bool operator>=(ParallelTensorShape const &) const;
  ::FlexFlow::ParallelTensorDims dims;
  ::FlexFlow::DataType data_type;
};
} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::ParallelTensorShape> {
  size_t operator()(FlexFlow::ParallelTensorShape const &) const;
};
} // namespace std

namespace nlohmann {
template <>
struct adl_serializer<FlexFlow::ParallelTensorShape> {
  static FlexFlow::ParallelTensorShape from_json(json const &);
  static void to_json(json &, FlexFlow::ParallelTensorShape const &);
};
} // namespace nlohmann

namespace FlexFlow {
std::string format_as(ParallelTensorShape const &);
std::ostream &operator<<(std::ostream &, ParallelTensorShape const &);
} // namespace FlexFlow

#endif // _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_TENSOR_SHAPE_DTG_H
