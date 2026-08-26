#include <doctest/doctest.h>
#include "pcg/mapped_parallel_computation_graph/mapped_standard_operator_task_group.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("MappedStandardOperatorTaskGroup constructor") {
    auto machine_coord = [&](nonnegative_int idx)
      -> MachineSpaceCoordinate
    {
      return MachineSpaceCoordinate{
        /*node_idx=*/idx,
        /*device_idx=*/0_n,
      };
    };

    auto op_binding = [&](nonnegative_int lhs_input_idx,
                          nonnegative_int rhs_input_idx,
                          nonnegative_int output_idx1,
                          nonnegative_int output_idx2)
      -> OperatorAtomicTaskShardBinding
    {
      return OperatorAtomicTaskShardBinding{
        std::map<TensorSlotName, ParallelTensorSpaceCoordinate>{
          {
            TensorSlotName::LHS_INPUT,
            ParallelTensorSpaceCoordinate{
              /*sum_component=*/0_n,
              /*discard_copy_component=*/0_n,
              /*shard_components=*/FFOrdered<nonnegative_int>{
                lhs_input_idx,
                0_n,
              },
            },
          },
          {
            TensorSlotName::RHS_INPUT,
            ParallelTensorSpaceCoordinate{
              /*sum_component=*/0_n,
              /*discard_copy_component=*/0_n,
              /*shard_components=*/FFOrdered<nonnegative_int>{
                0_n,
                rhs_input_idx,
              },
            },
          },
          {
            TensorSlotName::OUTPUT,
            ParallelTensorSpaceCoordinate{
              /*sum_component=*/0_n,
              /*discard_copy_component=*/0_n,
              /*shard_components=*/FFOrdered<nonnegative_int>{
                output_idx1,
                output_idx2,
              },
            },
          },
        }
      };
    };

    SUBCASE("relation is biunique") {
      bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> rel = {
        {machine_coord(0_n),  op_binding(0_n, 0_n, 0_n, 0_n)},
        {machine_coord(2_n),  op_binding(1_n, 1_n, 0_n, 1_n)},
        {machine_coord(4_n),  op_binding(2_n, 2_n, 1_n, 0_n)},
        {machine_coord(6_n),  op_binding(3_n, 3_n, 1_n, 1_n)},
      };

      MappedStandardOperatorTaskGroup result = MappedStandardOperatorTaskGroup{rel};
    }

    SUBCASE("relation k-unique for k > 1 for lhs input") {
      bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> rel = {
        {machine_coord(0_n),  op_binding(0_n, 0_n, 0_n, 0_n)},
        {machine_coord(2_n),  op_binding(0_n, 1_n, 0_n, 1_n)},
        {machine_coord(4_n),  op_binding(1_n, 2_n, 1_n, 0_n)},
        {machine_coord(6_n),  op_binding(1_n, 3_n, 1_n, 1_n)},
      };

      CHECK_THROWS(MappedStandardOperatorTaskGroup{rel});
   }

    SUBCASE("relation k-unique for k > 1 for rhs input") {
      bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> rel = {
        {machine_coord(0_n),  op_binding(0_n, 0_n, 0_n, 0_n)},
        {machine_coord(2_n),  op_binding(1_n, 0_n, 0_n, 1_n)},
        {machine_coord(4_n),  op_binding(2_n, 1_n, 1_n, 0_n)},
        {machine_coord(6_n),  op_binding(3_n, 1_n, 1_n, 1_n)},
      };

      CHECK_THROWS(MappedStandardOperatorTaskGroup{rel});
   }

    SUBCASE("relation k-unique for k > 1 for output") {
      bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> rel = {
        {machine_coord(0_n),  op_binding(0_n, 0_n, 0_n, 0_n)},
        {machine_coord(2_n),  op_binding(1_n, 1_n, 0_n, 0_n)},
        {machine_coord(4_n),  op_binding(2_n, 2_n, 1_n, 0_n)},
        {machine_coord(6_n),  op_binding(3_n, 3_n, 1_n, 0_n)},
      };

      CHECK_THROWS(MappedStandardOperatorTaskGroup{rel});
    }

    SUBCASE("relation k-unique for k > 1 for output") {
      bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> rel = {
        {machine_coord(0_n),  op_binding(0_n, 0_n, 0_n, 0_n)},
        {machine_coord(2_n),  op_binding(1_n, 1_n, 0_n, 0_n)},
        {machine_coord(4_n),  op_binding(2_n, 2_n, 1_n, 0_n)},
        {machine_coord(6_n),  op_binding(3_n, 3_n, 1_n, 0_n)},
      };

      CHECK_THROWS(MappedStandardOperatorTaskGroup{rel});
    }

    SUBCASE("relation just slightly left-unique") {
      bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> rel = {
        {machine_coord(0_n),  op_binding(0_n, 0_n, 0_n, 0_n)},
        {machine_coord(2_n),  op_binding(0_n, 1_n, 0_n, 1_n)},
        {machine_coord(4_n),  op_binding(1_n, 2_n, 1_n, 0_n)},
        {machine_coord(6_n),  op_binding(1_n, 3_n, 1_n, 1_n)},
        {machine_coord(8_n),  op_binding(1_n, 4_n, 1_n, 1_n)},
      };

      CHECK_THROWS(MappedStandardOperatorTaskGroup{rel});
    }
  }
}
