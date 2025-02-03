#include "./cost_estimator_for_test.h"
#include "compiler/cost_estimator/op_cost_metrics.dtg.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_tensor_set_movement.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/unmapped_op_cost_estimate_key.h"
#include "utils/nonnegative_int/nonnegative_int.h"

namespace FlexFlow {

TestCostEstimator::TestCostEstimator(
    std::function<OpCostMetrics(OpCostEstimateKey const &)> const
        &get_operator_cost,
    std::function<float(TensorSetMovement const &)> const
        &get_communication_cost)
    : get_operator_cost(get_operator_cost),
      get_communication_cost(get_communication_cost) {}

OpCostMetrics
    TestCostEstimator::estimate_cost(OpCostEstimateKey const &k) const {
  return this->get_operator_cost(k);
}

float TestCostEstimator::estimate_cost(TensorSetMovement const &m) const {
  return this->get_communication_cost(m);
}

CostEstimator make_fake_cost_estimator(
    std::function<OpCostMetrics(OpCostEstimateKey const &)> const
        &get_operator_cost,
    std::function<float(TensorSetMovement const &)> const
        &get_communication_cost) {
  return CostEstimator::create<TestCostEstimator>(get_operator_cost,
                                                  get_communication_cost);
}

CostEstimator make_fake_cost_estimator(
    std::unordered_map<OpCostEstimateKey, OpCostMetrics> const &op_cost_map,
    std::unordered_map<TensorSetMovement, float> const &comm_cost_map) {
  return make_fake_cost_estimator(
      [op_cost_map](OpCostEstimateKey const &k) { return op_cost_map.at(k); },
      [comm_cost_map](TensorSetMovement const &m) {
        return comm_cost_map.at(m);
      });
}

CostEstimator make_fake_constant_cost_estimator(float forward_op_cost,
                                                float backward_op_cost,
                                                float comm_cost,
                                                nonnegative_int memory_cost) {
  return make_fake_cost_estimator(
      [=](OpCostEstimateKey const &op) {
        return OpCostMetrics{forward_op_cost, backward_op_cost, memory_cost};
      },
      [=](TensorSetMovement const &op) { return comm_cost; });
}

}
