#include "pcg/mapped_parallel_computation_graph/mapped_parallelism_operator_task_group.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("MappedParallelismOperatorTaskGroup constructor") {
    auto machine_coord = [&](int idx) -> MachineSpaceCoordinate {
      return MachineSpaceCoordinate{
          /*node_idx=*/nonnegative_int{idx},
          /*device_idx=*/0_n,
      };
    };

    auto op_binding = [&](int input_idx, int output_idx, int output_idx2 = 0)
        -> ParallelismOperatorAtomicTaskShardBinding {
      return ParallelismOperatorAtomicTaskShardBinding{
          /*input_coord=*/ParallelTensorSpaceCoordinate{
              /*sum_component=*/0_n,
              /*discard_copy_component=*/0_n,
              /*shard_components=*/
              FFOrdered<nonnegative_int>{
                  nonnegative_int{input_idx},
                  0_n,
              },
          },
          /*output_coord=*/
          ParallelTensorSpaceCoordinate{
              /*sum_component=*/0_n,
              /*discard_copy_component=*/0_n,
              /*shard_components=*/
              FFOrdered<nonnegative_int>{
                  nonnegative_int{output_idx},
                  nonnegative_int{output_idx2},
              },
          },
      };
    };

    SUBCASE("relation k-unique for k > 1 for input") {
      bidict<MachineSpaceCoordinate, ParallelismOperatorAtomicTaskShardBinding>
          rel = {
              {machine_coord(0), op_binding(0, 0)},
              {machine_coord(2), op_binding(0, 1)},
              {machine_coord(4), op_binding(1, 2)},
              {machine_coord(6), op_binding(1, 3)},
          };

      MappedParallelismOperatorTaskGroup result =
          MappedParallelismOperatorTaskGroup{rel};
    }

    SUBCASE("relation k-unique for k > 1 for output") {
      bidict<MachineSpaceCoordinate, ParallelismOperatorAtomicTaskShardBinding>
          rel = {
              {machine_coord(0), op_binding(0, 0)},
              {machine_coord(2), op_binding(1, 0)},
              {machine_coord(4), op_binding(2, 0)},
              {machine_coord(6), op_binding(3, 1)},
              {machine_coord(8), op_binding(4, 1)},
              {machine_coord(10), op_binding(5, 1)},
          };

      MappedParallelismOperatorTaskGroup result =
          MappedParallelismOperatorTaskGroup{rel};
    }

    SUBCASE("relation k-unique for k > 1 and there are multiple output dims") {
      bidict<MachineSpaceCoordinate, ParallelismOperatorAtomicTaskShardBinding>
          rel = {
              {machine_coord(0), op_binding(0, 0, 0)},
              {machine_coord(2), op_binding(0, 0, 1)},
              {machine_coord(4), op_binding(0, 1, 0)},
              {machine_coord(6), op_binding(0, 1, 1)},
              {machine_coord(8), op_binding(1, 0, 0)},
              {machine_coord(10), op_binding(1, 0, 1)},
              {machine_coord(12), op_binding(1, 1, 0)},
              {machine_coord(14), op_binding(1, 1, 1)},
          };

      CHECK_THROWS(MappedParallelismOperatorTaskGroup{rel});
    }

    SUBCASE("relation is biunique") {
      bidict<MachineSpaceCoordinate, ParallelismOperatorAtomicTaskShardBinding>
          rel = {
              {machine_coord(0), op_binding(0, 0)},
              {machine_coord(2), op_binding(1, 1)},
              {machine_coord(4), op_binding(2, 2)},
              {machine_coord(6), op_binding(3, 3)},
          };

      CHECK_THROWS(MappedParallelismOperatorTaskGroup{rel});
    }

    SUBCASE("relation is not k-unique for any k") {
      SUBCASE("relation is left-unique with inconsistent cardinality") {
        bidict<MachineSpaceCoordinate,
               ParallelismOperatorAtomicTaskShardBinding>
            rel = {
                {machine_coord(0), op_binding(0, 0)},
                {machine_coord(2), op_binding(0, 1)},
                {machine_coord(4), op_binding(1, 2)},
                {machine_coord(6), op_binding(1, 3)},
                {machine_coord(8), op_binding(1, 4)},
            };

        CHECK_THROWS(MappedParallelismOperatorTaskGroup{rel});
      }

      SUBCASE("relation is right-unique with inconsistent cardinality") {
        bidict<MachineSpaceCoordinate,
               ParallelismOperatorAtomicTaskShardBinding>
            rel = {
                {machine_coord(0), op_binding(0, 0)},
                {machine_coord(2), op_binding(1, 0)},
                {machine_coord(4), op_binding(2, 0)},
                {machine_coord(6), op_binding(3, 1)},
                {machine_coord(8), op_binding(4, 1)},
            };

        CHECK_THROWS(MappedParallelismOperatorTaskGroup{rel});
      }

      SUBCASE("relation is not hemiunique") {
        SUBCASE("relation is universal") {
          bidict<MachineSpaceCoordinate,
                 ParallelismOperatorAtomicTaskShardBinding>
              rel = {
                  {machine_coord(0), op_binding(0, 0)},
                  {machine_coord(2), op_binding(0, 1)},
                  {machine_coord(4), op_binding(1, 0)},
                  {machine_coord(6), op_binding(1, 1)},
              };

          CHECK_THROWS(MappedParallelismOperatorTaskGroup{rel});
        }

        SUBCASE("relation is hemiunique-composite") {
          bidict<MachineSpaceCoordinate,
                 ParallelismOperatorAtomicTaskShardBinding>
              rel = {
                  {machine_coord(0), op_binding(0, 0)},
                  {machine_coord(2), op_binding(0, 1)},
                  {machine_coord(4), op_binding(1, 2)},
                  {machine_coord(6), op_binding(2, 2)},
              };

          CHECK_THROWS(MappedParallelismOperatorTaskGroup{rel});
        }
      }
    }
  }
}
