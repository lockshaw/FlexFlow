#include "op-attrs/operator_type.h"

namespace FlexFlow {

std::string get_operator_type_name(OperatorType op) {
  return fmt::to_string(op);
}

bool is_parallel_op(OperatorType t) {
  switch (t) {
    case OperatorType::REPARTITION:
    case OperatorType::COMBINE:
    case OperatorType::REPLICATE:
    case OperatorType::REDUCTION:
    case OperatorType::BATCH:
    case OperatorType::PIPELINE:
    case OperatorType::FUSED_PARALLEL:
      return true;
    default:
      return false;
  }
}

bool should_be_mapped(OperatorType t) {
  if (is_parallel_op(t)) {
    return false;
  }

  if (t == OperatorType::INPUT || t == OperatorType::WEIGHT) {
    return false;
  }

  return true;
}


} // namespace FlexFlow
