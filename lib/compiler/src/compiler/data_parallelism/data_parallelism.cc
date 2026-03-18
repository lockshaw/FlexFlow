#include "compiler/data_parallelism/data_parallelism.h"
#include "substitutions/operator_pattern/operator_attribute_constraint.h"
#include "substitutions/output_graph/output_operator_attrs_assignment.h"
#include "substitutions/substitution.dtg.h"
#include "substitutions/substitution_builder.h"
#include "substitutions/tensor_pattern/tensor_attribute_pattern.h"
#include "utils/containers/require_only_key.h"

namespace FlexFlow {

Substitution get_data_parallelism_substitution(int_ge_two degree) {
  SubstitutionBuilder b;

  OperatorAttributePattern p_input_pattern = OperatorAttributePattern{{
    op_type_equals_constraint(OperatorType::INPUT),
  }};

  std::string input_node_name = "input_node";
  PatternValue p_input_tensor = require_only_key(
    b.add_pattern_node(
      /*node_pattern=*/p_input_pattern,
      /*inputs=*/{},
      /*output_patterns=*/{
        {TensorSlotName::OUTPUT, tensor_attribute_pattern_match_all()},
      },
      /*name=*/input_node_name),
    TensorSlotName::OUTPUT);

  OutputOperatorAttrsAssignment o_input_node_expr = output_operator_clone_node(
    b.pattern_node_named(input_node_name));

  OutputGraphExprValue o_unpartitioned_input = require_only_key(
    b.add_output_graph_node(
      /*node_expr=*/o_input_node_expr,
      /*inputs=*/{},
      /*output_slots=*/{TensorSlotName::OUTPUT}),
    TensorSlotName::OUTPUT);

  OutputOperatorAttrsAssignment o_partition_node_expr =
    fixed_output_operator_attrs_assignment_for_partition(
      /*dim=*/ff_dim_t{0_n},
      /*degree=*/degree);

  OutputGraphExprValue o_partitioned_input = require_only_key(
    b.add_output_graph_node(
      /*node_expr=*/o_partition_node_expr,
      /*inputs=*/{{TensorSlotName::INPUT, o_unpartitioned_input}},
      /*output_slots=*/{TensorSlotName::OUTPUT}),
    TensorSlotName::OUTPUT);

  b.equate_outputs(p_input_tensor, o_partitioned_input);

  return b.get_substitution();
}

SearchResult
  apply_data_parallelism(ComputationGraph const &cg,
                         int_ge_two degree)
{
  NOT_IMPLEMENTED();
}


} // namespace FlexFlow
