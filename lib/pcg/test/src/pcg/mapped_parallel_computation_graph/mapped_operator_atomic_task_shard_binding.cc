#include <doctest/doctest.h>
#include "pcg/mapped_parallel_computation_graph/mapped_operator_atomic_task_shard_binding.h"
#include "utils/exception.h"
#include "test/utils/doctest/fmt/optional.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("mapped_op_task_shard_bindings_are_k_unique_on_slot") {
    auto ms_coord = [&](nonnegative_int idx) -> MachineSpaceCoordinate {
      return MachineSpaceCoordinate{
        /*node_idx=*/idx,
        /*device_idx=*/0_n,
      };
    };

    auto pt_coord = [&](nonnegative_int idx) -> ParallelTensorSpaceCoordinate {
      return ParallelTensorSpaceCoordinate{
        /*sum_component=*/0_n,
        /*discard_copy_component=*/0_n,
        /*shard_components=*/FFOrdered<nonnegative_int>{
          0_n,
          idx,
        },
      };
    };

    auto mk_binding = [&](nonnegative_int device_idx, 
                          nonnegative_int input_idx,
                          nonnegative_int output_idx)
      -> MappedOperatorAtomicTaskShardBinding
    {
      return MappedOperatorAtomicTaskShardBinding{
        /*tensor_coords=*/{
          {TensorSlotName::INPUT, pt_coord(input_idx)},
          {TensorSlotName::OUTPUT, pt_coord(output_idx)},
        },
        /*machine_coord=*/ms_coord(device_idx),
      };
    };

    SUBCASE("bindings are actually k-unique") {
      std::set<MappedOperatorAtomicTaskShardBinding> bindings = {
        mk_binding(0_n, 0_n, 0_n),
        mk_binding(1_n, 0_n, 1_n),
        mk_binding(2_n, 0_n, 2_n),
        mk_binding(3_n, 1_n, 2_n),
        mk_binding(4_n, 1_n, 3_n),
        mk_binding(5_n, 1_n, 4_n),
      };

      SUBCASE("on the slot in question") {
        std::optional<nonnegative_int> result = mapped_op_task_shard_bindings_are_k_unique_on_slot(
          bindings, TensorSlotName::INPUT);

        std::optional<nonnegative_int> correct = 3_n;

        CHECK(result == correct);
      }

      SUBCASE("but not the slot in question") {
        std::optional<nonnegative_int> result = mapped_op_task_shard_bindings_are_k_unique_on_slot(
          bindings, TensorSlotName::OUTPUT);

        std::optional<nonnegative_int> correct = std::nullopt;

        CHECK(result == correct);
      }
    }

    SUBCASE("bindings are not k-unique on any slot") {
      std::set<MappedOperatorAtomicTaskShardBinding> bindings = {
        mk_binding(0_n, 0_n, 0_n),
        mk_binding(1_n, 0_n, 1_n),
        mk_binding(2_n, 0_n, 2_n),
        mk_binding(3_n, 1_n, 2_n),
        mk_binding(4_n, 1_n, 3_n),
      };

      std::optional<nonnegative_int> result = mapped_op_task_shard_bindings_are_k_unique_on_slot(
        bindings, TensorSlotName::INPUT);

      std::optional<nonnegative_int> correct = std::nullopt;

      CHECK(result == correct);
    }
  }

  TEST_CASE("mapped_op_task_shard_bindings_are_unique_on_slot") {
    auto ms_coord = [&](nonnegative_int idx) -> MachineSpaceCoordinate {
      return MachineSpaceCoordinate{
        /*node_idx=*/idx,
        /*device_idx=*/0_n,
      };
    };

    auto pt_coord = [&](nonnegative_int idx) -> ParallelTensorSpaceCoordinate {
      return ParallelTensorSpaceCoordinate{
        /*sum_component=*/0_n,
        /*discard_copy_component=*/0_n,
        /*shard_components=*/FFOrdered<nonnegative_int>{
          0_n,
          idx,
        },
      };
    };

    auto mk_binding = [&](nonnegative_int device_idx, 
                          nonnegative_int input_idx,
                          nonnegative_int output_idx)
      -> MappedOperatorAtomicTaskShardBinding
    {
      return MappedOperatorAtomicTaskShardBinding{
        /*tensor_coords=*/{
          {TensorSlotName::INPUT, pt_coord(input_idx)},
          {TensorSlotName::OUTPUT, pt_coord(output_idx)},
        },
        /*machine_coord=*/ms_coord(device_idx),
      };
    };
    
    SUBCASE("bindings are actually unique") {
      std::set<MappedOperatorAtomicTaskShardBinding> bindings = {
        mk_binding(0_n, 0_n, 0_n),
        mk_binding(2_n, 1_n, 0_n),
        mk_binding(1_n, 2_n, 0_n),
        mk_binding(3_n, 3_n, 0_n),
        mk_binding(4_n, 4_n, 0_n),
        mk_binding(6_n, 5_n, 0_n),
      };

      SUBCASE("on the slot in question") {
        bool result = mapped_op_task_shard_bindings_are_unique_on_slot(bindings, TensorSlotName::INPUT);

        bool correct = true;

        CHECK(result == correct);
      }

      SUBCASE("but not on the slot in question") {
        bool result = mapped_op_task_shard_bindings_are_unique_on_slot(bindings, TensorSlotName::OUTPUT);

        bool correct = false;

        CHECK(result == correct);
      }
    }
    
    SUBCASE("bindings are not unique on any slot") {
      std::set<MappedOperatorAtomicTaskShardBinding> bindings = {
        mk_binding(0_n, 0_n, 0_n),
        mk_binding(2_n, 0_n, 0_n),
        mk_binding(1_n, 0_n, 0_n),
        mk_binding(3_n, 0_n, 0_n),
        mk_binding(4_n, 0_n, 0_n),
        mk_binding(6_n, 0_n, 0_n),
      };

      bool result = mapped_op_task_shard_bindings_are_unique_on_slot(bindings, TensorSlotName::INPUT);

      bool correct = false;

      CHECK(result == correct);
    }
  }

  TEST_CASE("mapped_op_task_shard_binding_project_out_key") {
    // TODO(@lockshaw)(#pr):
    NOT_IMPLEMENTED();
  }
}
