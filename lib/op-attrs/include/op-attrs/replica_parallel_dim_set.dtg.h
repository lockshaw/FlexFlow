// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/op-attrs/include/op-attrs/replica_parallel_dim_set.struct.toml
/* proj-data
{
  "generated_from": "20d8004e6f1e710688fe692b92dc2816"
}
*/

#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_REPLICA_PARALLEL_DIM_SET_DTG_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_REPLICA_PARALLEL_DIM_SET_DTG_H

#include "fmt/format.h"
#include "nlohmann/json.hpp"
#include "rapidcheck.h"
#include <functional>
#include <ostream>
#include <tuple>

namespace FlexFlow {
struct ReplicaParallelDimSet {
  ReplicaParallelDimSet() = delete;
  ReplicaParallelDimSet(int const &sum_degree, int const &discard_copy_degree);

  bool operator==(ReplicaParallelDimSet const &) const;
  bool operator!=(ReplicaParallelDimSet const &) const;
  bool operator<(ReplicaParallelDimSet const &) const;
  bool operator>(ReplicaParallelDimSet const &) const;
  bool operator<=(ReplicaParallelDimSet const &) const;
  bool operator>=(ReplicaParallelDimSet const &) const;
  int sum_degree;
  int discard_copy_degree;
};
} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::ReplicaParallelDimSet> {
  size_t operator()(FlexFlow::ReplicaParallelDimSet const &) const;
};
} // namespace std

namespace nlohmann {
template <>
struct adl_serializer<FlexFlow::ReplicaParallelDimSet> {
  static FlexFlow::ReplicaParallelDimSet from_json(json const &);
  static void to_json(json &, FlexFlow::ReplicaParallelDimSet const &);
};
} // namespace nlohmann

namespace rc {
template <>
struct Arbitrary<FlexFlow::ReplicaParallelDimSet> {
  static Gen<FlexFlow::ReplicaParallelDimSet> arbitrary();
};
} // namespace rc

namespace FlexFlow {
std::string format_as(ReplicaParallelDimSet const &);
std::ostream &operator<<(std::ostream &, ReplicaParallelDimSet const &);
} // namespace FlexFlow

#endif // _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_REPLICA_PARALLEL_DIM_SET_DTG_H
