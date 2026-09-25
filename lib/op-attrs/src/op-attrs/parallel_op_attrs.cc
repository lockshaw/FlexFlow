#include "op-attrs/parallel_op_attrs.h"
#include "op-attrs/ops/combine.h"
#include "op-attrs/ops/reduction.h"
#include "op-attrs/ops/repartition.h"
#include "op-attrs/ops/replicate.h"
#include "utils/overload.h"

namespace FlexFlow {

ParallelTensorShape parallel_op_attrs_get_output_parallel_shape(ParallelOpAttrs const &attrs,
                                     ParallelTensorShape const &input_shape) {
  return attrs.visit<ParallelTensorShape>(overload{
      [&](CombineAttrs const &combine_attrs) {
        return combine_get_output_parallel_shape(combine_attrs, input_shape);
      },
      [&](ReductionAttrs const &reduction_attrs) {
        return reduction_get_output_parallel_shape(reduction_attrs, input_shape);
      },
      [&](RepartitionAttrs const &repartition_attrs) {
        return repartition_get_output_parallel_shape(repartition_attrs,
                                                     input_shape);
      },
      [&](ReplicateAttrs const &replicate_attrs) {
        return replicate_get_output_parallel_shape(replicate_attrs, input_shape);
      },
  });
}

PCGOperatorAttrs
    pcg_op_attrs_from_parallel_op_attrs(ParallelOpAttrs const &attrs) {
  return attrs.visit<PCGOperatorAttrs>(
      [](auto const &attrs) { return PCGOperatorAttrs{attrs}; });
}

std::optional<ParallelOpAttrs>
    parallel_op_attrs_from_pcg_op_attrs(PCGOperatorAttrs const &op)
{
  return op.visit<std::optional<ParallelOpAttrs>>(overload{
      [&](CombineAttrs const &attrs) { return ParallelOpAttrs{attrs}; },
      [&](ReductionAttrs const &attrs) { return ParallelOpAttrs{attrs}; },
      [&](RepartitionAttrs const &attrs) { return ParallelOpAttrs{attrs}; },
      [&](ReplicateAttrs const &attrs) { return ParallelOpAttrs{attrs}; },
      [](auto const &attrs) { return std::nullopt; },
  });
}

} // namespace FlexFlow
