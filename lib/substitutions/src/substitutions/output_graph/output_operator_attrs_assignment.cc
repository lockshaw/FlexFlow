#include "substitutions/output_graph/output_operator_attrs_assignment.h"
#include "substitutions/output_graph/materialize_operator_from_attrs_map.h"
#include "substitutions/output_graph/output_operator_attribute_expr.h"
#include "utils/containers/map_values.h"

namespace FlexFlow {

OutputOperatorAttrsAssignment output_operator_clone_node(PatternNode const &) {
  NOT_IMPLEMENTED();
}

PCGOperatorAttrs materialize_output_operator_from_attrs_assignment(OutputOperatorAttrsAssignment const &attrs_assignment,
                                                                   std::unordered_map<PatternNode, PCGOperatorAttrs> const &node_match) {
  std::unordered_map<OperatorAttributeKey, OperatorAttributeValue> attr_map = map_values(
    attrs_assignment.assignments,
    [&](OutputOperatorAttributeExpr const &expr) { 
      return evaluate_output_operator_attribute_expr(expr, node_match);
    }
  );

  return materialize_operator_from_attrs_map(attr_map);
}

} // namespace FlexFlow
