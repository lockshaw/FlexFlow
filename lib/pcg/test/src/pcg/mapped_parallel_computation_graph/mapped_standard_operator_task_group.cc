#include "pcg/mapped_parallel_computation_graph/mapped_standard_operator_task_group.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("MappedStandardOperatorTaskGroup constructor") {
    auto machine_coord = [&](int idx) -> MachineSpaceCoordinate {
      return MachineSpaceCoordinate{
          /*node_idx=*/nonnegative_int{idx},
          /*device_idx=*/0_n,
      };
    };

    auto op_binding = [&](int lhs_input_idx,
                          int rhs_input_idx,
                          int output_idx1,
                          int output_idx2) -> OperatorAtomicTaskShardBinding {
      return OperatorAtomicTaskShardBinding{
          std::map<TensorSlotName, ParallelTensorSpaceCoordinate>{
              {
                  TensorSlotName::LHS_INPUT,
                  ParallelTensorSpaceCoordinate{
                      /*sum_component=*/0_n,
                      /*discard_copy_component=*/0_n,
                      /*shard_components=*/
                      FFOrdered<nonnegative_int>{
                          nonnegative_int{lhs_input_idx},
                          0_n,
                      },
                  },
              },
              {
                  TensorSlotName::RHS_INPUT,
                  ParallelTensorSpaceCoordinate{
                      /*sum_component=*/0_n,
                      /*discard_copy_component=*/0_n,
                      /*shard_components=*/
                      FFOrdered<nonnegative_int>{
                          0_n,
                          nonnegative_int{rhs_input_idx},
                      },
                  },
              },
              {
                  TensorSlotName::OUTPUT,
                  ParallelTensorSpaceCoordinate{
                      /*sum_component=*/0_n,
                      /*discard_copy_component=*/0_n,
                      /*shard_components=*/
                      FFOrdered<nonnegative_int>{
                          nonnegative_int{output_idx1},
                          nonnegative_int{output_idx2},
                      },
                  },
              },
          }};
    };

    SUBCASE("relation is biunique") {
      bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> rel = {
          {machine_coord(0), op_binding(0, 0, 0, 0)},
          {machine_coord(2), op_binding(1, 1, 0, 1)},
          {machine_coord(4), op_binding(2, 2, 1, 0)},
          {machine_coord(6), op_binding(3, 3, 1, 1)},
      };

      MappedStandardOperatorTaskGroup result =
          MappedStandardOperatorTaskGroup{rel};
    }

    SUBCASE("relation k-unique for k > 1 for lhs input") {
      bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> rel = {
          {machine_coord(0), op_binding(0, 0, 0, 0)},
          {machine_coord(2), op_binding(0, 1, 0, 1)},
          {machine_coord(4), op_binding(1, 2, 1, 0)},
          {machine_coord(6), op_binding(1, 3, 1, 1)},
      };

      CHECK_THROWS(MappedStandardOperatorTaskGroup{rel});
    }

    SUBCASE("relation k-unique for k > 1 for rhs input") {
      bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> rel = {
          {machine_coord(0), op_binding(0, 0, 0, 0)},
          {machine_coord(2), op_binding(1, 0, 0, 1)},
          {machine_coord(4), op_binding(2, 1, 1, 0)},
          {machine_coord(6), op_binding(3, 1, 1, 1)},
      };

      CHECK_THROWS(MappedStandardOperatorTaskGroup{rel});
    }

    SUBCASE("relation k-unique for k > 1 for output") {
      bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> rel = {
          {machine_coord(0), op_binding(0, 0, 0, 0)},
          {machine_coord(2), op_binding(1, 1, 0, 0)},
          {machine_coord(4), op_binding(2, 2, 1, 0)},
          {machine_coord(6), op_binding(3, 3, 1, 0)},
      };

      CHECK_THROWS(MappedStandardOperatorTaskGroup{rel});
    }

    SUBCASE("relation k-unique for k > 1 for output") {
      bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> rel = {
          {machine_coord(0), op_binding(0, 0, 0, 0)},
          {machine_coord(2), op_binding(1, 1, 0, 0)},
          {machine_coord(4), op_binding(2, 2, 1, 0)},
          {machine_coord(6), op_binding(3, 3, 1, 0)},
      };

      CHECK_THROWS(MappedStandardOperatorTaskGroup{rel});
    }

    SUBCASE("relation just slightly left-unique") {
      bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> rel = {
          {machine_coord(0), op_binding(0, 0, 0, 0)},
          {machine_coord(2), op_binding(0, 1, 0, 1)},
          {machine_coord(4), op_binding(1, 2, 1, 0)},
          {machine_coord(6), op_binding(1, 3, 1, 1)},
          {machine_coord(8), op_binding(1, 4, 1, 1)},
      };

      CHECK_THROWS(MappedStandardOperatorTaskGroup{rel});
    }
  }
}
