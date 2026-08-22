#include "pcg/mapped_parallel_computation_graph/mapped_operator_task_group.h"
#include "op-attrs/parallel_tensor_space_coordinate.dtg.h"
#include "op-attrs/tensor_slot_name.dtg.h"
#include "pcg/device_type.dtg.h"
#include "pcg/machine_space_coordinate.dtg.h"
#include "pcg/mapped_parallel_computation_graph/operator_atomic_task_shard_binding.dtg.h"
#include <doctest/doctest.h>
#include <nlohmann/json.hpp>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("MappedOperatorTaskGroup constructor") {

    auto machine_coord = [&](nonnegative_int idx)
      -> MachineSpaceCoordinate
    {
      return MachineSpaceCoordinate{
        /*node_idx=*/idx,
        /*device_idx=*/0_n,
      };
    };

    auto three_slot_binding = [&](TensorSlotName lhs_slot_name,
                                  nonnegative_int lhs_input_idx,
                                  TensorSlotName rhs_slot_name,
                                  nonnegative_int rhs_input_idx,
                                  TensorSlotName output_slot_name,
                                  nonnegative_int output_idx)
      -> OperatorAtomicTaskShardBinding
    {
      return OperatorAtomicTaskShardBinding{
        std::map<TensorSlotName, ParallelTensorSpaceCoordinate>{
          {
            lhs_slot_name,
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
            rhs_slot_name,
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
            output_slot_name,
            ParallelTensorSpaceCoordinate{
              /*sum_component=*/0_n,
              /*discard_copy_component=*/0_n,
              /*shard_components=*/FFOrdered<nonnegative_int>{
                output_idx,
                0_n,
              },
            },
          },
        }
      };
    };

    auto two_slot_binding = [&](TensorSlotName input_slot_name,
                              nonnegative_int input_idx,
                              TensorSlotName output_slot_name,
                              nonnegative_int output_idx)
      -> OperatorAtomicTaskShardBinding
    {
      return OperatorAtomicTaskShardBinding{
        std::map<TensorSlotName, ParallelTensorSpaceCoordinate>{
          {
            input_slot_name,
            ParallelTensorSpaceCoordinate{
              /*sum_component=*/0_n,
              /*discard_copy_component=*/0_n,
              /*shard_components=*/FFOrdered<nonnegative_int>{
                input_idx,
                0_n,
              },
            },
          },
          {
            output_slot_name,
            ParallelTensorSpaceCoordinate{
              /*sum_component=*/0_n,
              /*discard_copy_component=*/0_n,
              /*shard_components=*/FFOrdered<nonnegative_int>{
                output_idx,
                0_n,
              },
            },
          },
        }
      };
    };

    SUBCASE("slots are INPUT and OUTPUT") {
      auto op_binding = [&](nonnegative_int input_idx, nonnegative_int output_idx)
        -> OperatorAtomicTaskShardBinding
      {
        return two_slot_binding(
          TensorSlotName::INPUT,
          input_idx,
          TensorSlotName::OUTPUT,
          output_idx);
      };

      SUBCASE("relation k-unique for k > 1 for input") {
        bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> rel = {
          {machine_coord(0_n),  op_binding(0_n, 0_n)},
          {machine_coord(2_n),  op_binding(0_n, 1_n)},
          {machine_coord(4_n),  op_binding(1_n, 2_n)},
          {machine_coord(6_n),  op_binding(1_n, 3_n)},
        };

        MappedOperatorTaskGroup result = MappedOperatorTaskGroup{rel};
      }

      SUBCASE("relation k-unique for k > 1 for output") {
        bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> rel = {
          {machine_coord(0_n),  op_binding(0_n, 0_n)},
          {machine_coord(2_n),  op_binding(1_n, 0_n)},
          {machine_coord(4_n),  op_binding(2_n, 1_n)},
          {machine_coord(6_n),  op_binding(3_n, 1_n)},
        };

        MappedOperatorTaskGroup result = MappedOperatorTaskGroup{rel};
      }

      SUBCASE("relation is biunique") {
        bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> rel = {
          {machine_coord(0_n),  op_binding(0_n, 0_n)},
          {machine_coord(2_n),  op_binding(1_n, 1_n)},
          {machine_coord(4_n),  op_binding(2_n, 2_n)},
          {machine_coord(6_n),  op_binding(3_n, 3_n)},
        };

        MappedOperatorTaskGroup result = MappedOperatorTaskGroup{rel};
      }

      SUBCASE("relation is right-unique, but not with consistent cardinality") {
        bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> rel = {
          {machine_coord(0_n),  op_binding(0_n, 0_n)},
          {machine_coord(2_n),  op_binding(1_n, 0_n)},
          {machine_coord(4_n),  op_binding(2_n, 1_n)},
          {machine_coord(6_n),  op_binding(3_n, 1_n)},
          {machine_coord(8_n),  op_binding(4_n, 1_n)},
        };

        CHECK_THROWS(MappedOperatorTaskGroup{rel});
      }

      SUBCASE("relation is left-unique, but not with consistent cardinality") {
        bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> rel = {
          {machine_coord(0_n),  op_binding(0_n, 0_n)},
          {machine_coord(2_n),  op_binding(0_n, 1_n)},
          {machine_coord(4_n),  op_binding(1_n, 2_n)},
          {machine_coord(6_n),  op_binding(1_n, 3_n)},
          {machine_coord(8_n),  op_binding(1_n, 4_n)},
        };

        CHECK_THROWS(MappedOperatorTaskGroup{rel});
      }

      SUBCASE("relation is universal") {
        bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> rel = {
          {machine_coord(0_n),  op_binding(0_n, 0_n)},
          {machine_coord(2_n),  op_binding(0_n, 1_n)},
          {machine_coord(6_n),  op_binding(1_n, 0_n)},
          {machine_coord(8_n),  op_binding(1_n, 1_n)},
        };

        CHECK_THROWS(MappedOperatorTaskGroup{rel});
      }
    }

    SUBCASE("slots are LHS_INPUT and OUTPUT") {
      auto op_binding = [&](nonnegative_int input_idx, nonnegative_int output_idx)
        -> OperatorAtomicTaskShardBinding
      {
        return two_slot_binding(
          TensorSlotName::LHS_INPUT,
          input_idx,
          TensorSlotName::OUTPUT,
          output_idx);
      };

      SUBCASE("relation k-unique for k > 1 for input") {
        bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> rel = {
          {machine_coord(0_n),  op_binding(0_n, 0_n)},
          {machine_coord(2_n),  op_binding(0_n, 1_n)},
          {machine_coord(4_n),  op_binding(1_n, 2_n)},
          {machine_coord(6_n),  op_binding(1_n, 3_n)},
        };

        CHECK_THROWS(MappedOperatorTaskGroup{rel});
      }

      SUBCASE("relation k-unique for k > 1 for output") {
        bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> rel = {
          {machine_coord(0_n),  op_binding(0_n, 0_n)},
          {machine_coord(2_n),  op_binding(1_n, 0_n)},
          {machine_coord(4_n),  op_binding(2_n, 1_n)},
          {machine_coord(6_n),  op_binding(3_n, 1_n)},
        };

        CHECK_THROWS(MappedOperatorTaskGroup{rel});
      }

      SUBCASE("relation is biunique") {
        bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> rel = {
          {machine_coord(0_n),  op_binding(0_n, 0_n)},
          {machine_coord(2_n),  op_binding(1_n, 1_n)},
          {machine_coord(4_n),  op_binding(2_n, 2_n)},
          {machine_coord(6_n),  op_binding(3_n, 3_n)},
        };

        MappedOperatorTaskGroup result = MappedOperatorTaskGroup{rel};
      }
    }

    SUBCASE("slots are LHS_INPUT, RHS_INPUT, and OUTPUT") {
      auto op_binding = [&](nonnegative_int lhs_input_idx,
                            nonnegative_int rhs_input_idx,
                            nonnegative_int output_idx)
        -> OperatorAtomicTaskShardBinding
      {
        return three_slot_binding(
          TensorSlotName::LHS_INPUT,
          lhs_input_idx,
          TensorSlotName::RHS_INPUT,
          rhs_input_idx,
          TensorSlotName::OUTPUT,
          output_idx);
      };

      SUBCASE("relation k-unique for k > 1 for lhs input") {
        bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> rel = {
          {machine_coord(0_n),  op_binding(0_n, 0_n, 0_n)},
          {machine_coord(2_n),  op_binding(0_n, 1_n, 1_n)},
          {machine_coord(4_n),  op_binding(1_n, 2_n, 2_n)},
          {machine_coord(6_n),  op_binding(1_n, 3_n, 3_n)},
        };

        CHECK_THROWS(MappedOperatorTaskGroup{rel});
      }

      SUBCASE("relation k-unique for k > 1 for rhs input") {
        bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> rel = {
          {machine_coord(0_n),  op_binding(0_n, 0_n, 0_n)},
          {machine_coord(2_n),  op_binding(1_n, 0_n, 1_n)},
          {machine_coord(4_n),  op_binding(2_n, 1_n, 2_n)},
          {machine_coord(6_n),  op_binding(3_n, 1_n, 3_n)},
        };

        CHECK_THROWS(MappedOperatorTaskGroup{rel});
      }

      SUBCASE("relation k-unique for k > 1 for output") {
        bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> rel = {
          {machine_coord(0_n),  op_binding(0_n, 0_n, 0_n)},
          {machine_coord(2_n),  op_binding(1_n, 1_n, 0_n)},
          {machine_coord(4_n),  op_binding(2_n, 2_n, 1_n)},
          {machine_coord(6_n),  op_binding(3_n, 3_n, 1_n)},
        };

        CHECK_THROWS(MappedOperatorTaskGroup{rel});
      }

      SUBCASE("relation is biunique") {
        bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> rel = {
          {machine_coord(0_n),  op_binding(0_n, 0_n, 0_n)},
          {machine_coord(2_n),  op_binding(1_n, 1_n, 1_n)},
          {machine_coord(4_n),  op_binding(2_n, 2_n, 2_n)},
          {machine_coord(6_n),  op_binding(3_n, 3_n, 3_n)},
        };

        MappedOperatorTaskGroup result = MappedOperatorTaskGroup{rel};
      }
    }
  }

  TEST_CASE("adl_serializer<MappedOperatorTaskGroup>") {
    bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding>
        shard_bindings{
            {
                MachineSpaceCoordinate{0_n, 0_n},
                OperatorAtomicTaskShardBinding{
                    {
                        {
                            TensorSlotName::INPUT,
                            ParallelTensorSpaceCoordinate{
                                /*sum_component=*/0_n,
                                /*discard_copy_component=*/0_n,
                                /*shard_components=*/FFOrdered{1_n, 2_n, 3_n},
                            },
                        },
                    },
                },
            },
        };
    MappedOperatorTaskGroup deserialized{shard_bindings};
    nlohmann::json serialized = shard_bindings;

    SUBCASE("to_json") {
      nlohmann::json result = deserialized;
      nlohmann::json correct = serialized;

      CHECK(result == correct);
    }

    SUBCASE("from_json") {
      MappedOperatorTaskGroup result = serialized;
      MappedOperatorTaskGroup correct = deserialized;

      CHECK(result == correct);
    }
  }
}
