#include "pcg/mapped_parallel_computation_graph/mapped_parallelism_operator_task_group.h"
#include "utils/exception.h"
#include "utils/hash-utils.h"
#include "utils/hash/tuple.h"

namespace FlexFlow {

MappedParallelismOperatorTaskGroup::MappedParallelismOperatorTaskGroup(
    bidict<MachineSpaceCoordinate, ParallelismOperatorAtomicTaskShardBinding> const
        &shard_bindings)
    : shard_bindings(shard_bindings) {
  NOT_IMPLEMENTED();
}

bool MappedParallelismOperatorTaskGroup::operator==(
    MappedParallelismOperatorTaskGroup const &other) const {
  return this->tie() == other.tie();
}

bool MappedParallelismOperatorTaskGroup::operator!=(
    MappedParallelismOperatorTaskGroup const &other) const {
  return this->tie() != other.tie();
}

bool MappedParallelismOperatorTaskGroup::operator<(
    MappedParallelismOperatorTaskGroup const &other) const {
  return this->tie() < other.tie();
}

bool MappedParallelismOperatorTaskGroup::operator>(
    MappedParallelismOperatorTaskGroup const &other) const {
  return this->tie() > other.tie();
}

bool MappedParallelismOperatorTaskGroup::operator<=(
    MappedParallelismOperatorTaskGroup const &other) const {
  return this->tie() <= other.tie();
}

bool MappedParallelismOperatorTaskGroup::operator>=(
    MappedParallelismOperatorTaskGroup const &other) const {
  return this->tie() >= other.tie();
}

std::tuple<
    bidict<MachineSpaceCoordinate, ParallelismOperatorAtomicTaskShardBinding> const &>
    MappedParallelismOperatorTaskGroup::tie() const {

  return std::tie(this->shard_bindings);
}

bidict<MachineSpaceCoordinate, ParallelismOperatorAtomicTaskShardBinding> const &
    MappedParallelismOperatorTaskGroup::get_shard_bindings() const {
  return this->shard_bindings;
}

nlohmann::json format_as(::FlexFlow::MappedParallelismOperatorTaskGroup const &m) {
  return m;
}

std::ostream &operator<<(std::ostream &s,
                         ::FlexFlow::MappedParallelismOperatorTaskGroup const &x) {
  return (s << fmt::to_string(x));
}

} // namespace FlexFlow

namespace std {

size_t hash<::FlexFlow::MappedParallelismOperatorTaskGroup>::operator()(
    ::FlexFlow::MappedParallelismOperatorTaskGroup const &x) const {
  return ::FlexFlow::get_std_hash(x.tie());
}

} // namespace std

namespace nlohmann {

::FlexFlow::MappedParallelismOperatorTaskGroup
    adl_serializer<::FlexFlow::MappedParallelismOperatorTaskGroup>::from_json(
        json const &j) {
  return ::FlexFlow::MappedParallelismOperatorTaskGroup{j.template get<
      ::FlexFlow::bidict<::FlexFlow::MachineSpaceCoordinate,
                         ::FlexFlow::ParallelismOperatorAtomicTaskShardBinding>>()};
}

void adl_serializer<::FlexFlow::MappedParallelismOperatorTaskGroup>::to_json(
    json &j, ::FlexFlow::MappedParallelismOperatorTaskGroup const &t) {
  j = t.get_shard_bindings();
}

} // namespace nlohmann
