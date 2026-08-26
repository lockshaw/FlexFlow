#include <doctest/doctest.h>
#include "pcg/mapped_parallel_computation_graph/mapped_parallelism_operator_task_group.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("MappedParallelismOperatorTaskGroup constructor") {
    auto machine_coord = [&](nonnegative_int idx)
      -> MachineSpaceCoordinate
    {
      return MachineSpaceCoordinate{
        /*node_idx=*/idx,
        /*device_idx=*/0_n,
      };
    };

    auto op_binding = [&](nonnegative_int input_idx,
                          nonnegative_int output_idx,
                          nonnegative_int output_idx2 = 0_n)
      -> ParallelismOperatorAtomicTaskShardBinding
    {
      return ParallelismOperatorAtomicTaskShardBinding{
        /*input_coord=*/ParallelTensorSpaceCoordinate{
          /*sum_component=*/0_n,
          /*discard_copy_component=*/0_n,
          /*shard_components=*/FFOrdered<nonnegative_int>{
            input_idx,
            0_n,
          },
        },
        /*output_coord=*/ParallelTensorSpaceCoordinate{
          /*sum_component=*/0_n,
          /*discard_copy_component=*/0_n,
          /*shard_components=*/FFOrdered<nonnegative_int>{
            output_idx,
            output_idx2,
          },
        },
      };
    };

    SUBCASE("relation k-unique for k > 1 for input") {
      bidict<MachineSpaceCoordinate, ParallelismOperatorAtomicTaskShardBinding> rel = {
        {machine_coord(0_n),  op_binding(0_n, 0_n)},
        {machine_coord(2_n),  op_binding(0_n, 1_n)},
        {machine_coord(4_n),  op_binding(1_n, 2_n)},
        {machine_coord(6_n),  op_binding(1_n, 3_n)},
      };

      MappedParallelismOperatorTaskGroup result = MappedParallelismOperatorTaskGroup{rel};
   }

    SUBCASE("relation k-unique for k > 1 for output") {
      bidict<MachineSpaceCoordinate, ParallelismOperatorAtomicTaskShardBinding> rel = {
        {machine_coord(0_n),  op_binding(0_n, 0_n)},
        {machine_coord(2_n),  op_binding(1_n, 0_n)},
        {machine_coord(4_n),  op_binding(2_n, 0_n)},
        {machine_coord(6_n),  op_binding(3_n, 1_n)},
        {machine_coord(8_n),  op_binding(4_n, 1_n)},
        {machine_coord(10_n), op_binding(5_n, 1_n)},
      };

      MappedParallelismOperatorTaskGroup result = MappedParallelismOperatorTaskGroup{rel};
    }

    SUBCASE("relation k-unique for k > 1 and there are multiple output dims") {
      bidict<MachineSpaceCoordinate, ParallelismOperatorAtomicTaskShardBinding> rel = {
        {machine_coord(0_n),  op_binding(0_n, 0_n, 0_n)},
        {machine_coord(2_n),  op_binding(0_n, 0_n, 1_n)},
        {machine_coord(4_n),  op_binding(0_n, 1_n, 0_n)},
        {machine_coord(6_n),  op_binding(0_n, 1_n, 1_n)},
        {machine_coord(8_n),  op_binding(1_n, 0_n, 0_n)},
        {machine_coord(10_n), op_binding(1_n, 0_n, 1_n)},
        {machine_coord(12_n), op_binding(1_n, 1_n, 0_n)},
        {machine_coord(14_n), op_binding(1_n, 1_n, 1_n)},
      };

      CHECK_THROWS(MappedParallelismOperatorTaskGroup{rel});
    }

    SUBCASE("relation is biunique") {
      bidict<MachineSpaceCoordinate, ParallelismOperatorAtomicTaskShardBinding> rel = {
        {machine_coord(0_n),  op_binding(0_n, 0_n)},
        {machine_coord(2_n),  op_binding(1_n, 1_n)},
        {machine_coord(4_n),  op_binding(2_n, 2_n)},
        {machine_coord(6_n),  op_binding(3_n, 3_n)},
      };

      CHECK_THROWS(MappedParallelismOperatorTaskGroup{rel});
    }

    SUBCASE("relation is not k-unique for any k") {
      SUBCASE("relation is left-unique with inconsistent cardinality") {
        bidict<MachineSpaceCoordinate, ParallelismOperatorAtomicTaskShardBinding> rel = {
          {machine_coord(0_n),  op_binding(0_n, 0_n)},
          {machine_coord(2_n),  op_binding(0_n, 1_n)},
          {machine_coord(4_n),  op_binding(1_n, 2_n)},
          {machine_coord(6_n),  op_binding(1_n, 3_n)},
          {machine_coord(8_n),  op_binding(1_n, 4_n)},
        };

        CHECK_THROWS(MappedParallelismOperatorTaskGroup{rel});
      }

      SUBCASE("relation is right-unique with inconsistent cardinality") {
        bidict<MachineSpaceCoordinate, ParallelismOperatorAtomicTaskShardBinding> rel = {
          {machine_coord(0_n),  op_binding(0_n, 0_n)},
          {machine_coord(2_n),  op_binding(1_n, 0_n)},
          {machine_coord(4_n),  op_binding(2_n, 0_n)},
          {machine_coord(6_n),  op_binding(3_n, 1_n)},
          {machine_coord(8_n),  op_binding(4_n, 1_n)},
        };

        CHECK_THROWS(MappedParallelismOperatorTaskGroup{rel});
      }

      SUBCASE("relation is not hemiunique") {
        SUBCASE("relation is universal") {
          bidict<MachineSpaceCoordinate, ParallelismOperatorAtomicTaskShardBinding> rel = {
            {machine_coord(0_n),  op_binding(0_n, 0_n)},
            {machine_coord(2_n),  op_binding(0_n, 1_n)},
            {machine_coord(4_n),  op_binding(1_n, 0_n)},
            {machine_coord(6_n),  op_binding(1_n, 1_n)},
          };

          CHECK_THROWS(MappedParallelismOperatorTaskGroup{rel});
        }

        SUBCASE("relation is hemiunique-composite") {
          bidict<MachineSpaceCoordinate, ParallelismOperatorAtomicTaskShardBinding> rel = {
            {machine_coord(0_n),  op_binding(0_n, 0_n)},
            {machine_coord(2_n),  op_binding(0_n, 1_n)},
            {machine_coord(4_n),  op_binding(1_n, 2_n)},
            {machine_coord(6_n),  op_binding(2_n, 2_n)},
          };

          CHECK_THROWS(MappedParallelismOperatorTaskGroup{rel});
        }
      }
    }
  }
}
