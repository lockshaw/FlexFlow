#include "op-attrs/computation_graph_op_attrs.h"
#include "op-attrs/get_op_type.h"
#include "op-attrs/ops/broadcast.h"
#include "op-attrs/ops/cast.h"
#include "op-attrs/ops/embedding.h"
#include "op-attrs/ops/linear.h"
#include "op-attrs/ops/weight.h"
#include "utils/overload.h"

namespace FlexFlow {

OperatorType get_op_type(ComputationGraphOpAttrs const &attrs) {
  return attrs.visit<OperatorType>(
      [](auto const &x) { return get_op_type(x); });
}

RecordFormatter cg_op_attrs_as_dot(ComputationGraphOpAttrs const &attrs) {
  RecordFormatter attrs_record = attrs.visit<RecordFormatter>(overload{
      [](LinearAttrs const &l) { return as_dot(l); },
      [](CastAttrs const &a) { return as_dot(a); },
      [](EmbeddingAttrs const &a) { return as_dot(a); },
      [](WeightAttrs const &a) { return as_dot(a); },
      [](BroadcastAttrs const &a) { return as_dot(a); },
      [&](auto const &) { return mk_empty_record(Orientation::HORIZONTAL); },
  });

  RecordFormatter rr = mk_empty_record(Orientation::HORIZONTAL);
  rr << "Op Type" << fmt::to_string(get_op_type(attrs));

  RecordFormatter result = mk_empty_record(Orientation::VERTICAL);
  result << rr;
  result << attrs_record;

  return result;
}

std::optional<ComputationGraphOpAttrs>
    compgraph_op_attrs_from_pcg_op_attrs(PCGOperatorAttrs const &op) {

  return op.visit<std::optional<ComputationGraphOpAttrs>>(overload{
      [&](CombineAttrs const &attrs) { return std::nullopt; },
      [&](ReductionAttrs const &attrs) { return std::nullopt; },
      [&](RepartitionAttrs const &attrs) { return std::nullopt; },
      [&](ReplicateAttrs const &attrs) { return std::nullopt; },
      [](auto const &attrs) { return ComputationGraphOpAttrs{attrs}; },
  });
}

} // namespace FlexFlow
