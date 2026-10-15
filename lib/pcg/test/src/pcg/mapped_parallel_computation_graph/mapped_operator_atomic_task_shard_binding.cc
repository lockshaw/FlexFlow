#include "pcg/mapped_parallel_computation_graph/mapped_operator_atomic_task_shard_binding.h"
#include "test/utils/doctest/fmt/optional.h"
#include "test/utils/doctest/fmt/pair.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("mapped_op_task_shard_bindings_are_k_unique_on_slot") {
    auto ms_coord = [&](int idx) -> MachineSpaceCoordinate {
      return MachineSpaceCoordinate{
          /*node_idx=*/nonnegative_int{idx},
          /*device_idx=*/0_n,
      };
    };

    auto pt_coord = [&](int idx) -> ParallelTensorSpaceCoordinate {
      return ParallelTensorSpaceCoordinate{
          /*sum_component=*/0_n,
          /*discard_copy_component=*/0_n,
          /*shard_components=*/
          FFOrdered<nonnegative_int>{
              0_n,
              nonnegative_int{idx},
          },
      };
    };

    auto mk_binding = [&](int device_idx, int input_idx, int output_idx)
        -> MappedOperatorAtomicTaskShardBinding {
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
          mk_binding(0, 0, 0),
          mk_binding(1, 0, 1),
          mk_binding(2, 0, 2),
          mk_binding(3, 1, 2),
          mk_binding(4, 1, 3),
          mk_binding(5, 1, 4),
      };

      SUBCASE("on the slot in question") {
        std::optional<nonnegative_int> result =
            mapped_op_task_shard_bindings_are_k_unique_on_slot(
                bindings, TensorSlotName::INPUT);

        std::optional<nonnegative_int> correct = 3_n;

        CHECK(result == correct);
      }

      SUBCASE("but not the slot in question") {
        std::optional<nonnegative_int> result =
            mapped_op_task_shard_bindings_are_k_unique_on_slot(
                bindings, TensorSlotName::OUTPUT);

        std::optional<nonnegative_int> correct = std::nullopt;

        CHECK(result == correct);
      }
    }

    SUBCASE("bindings are not k-unique on any slot") {
      std::set<MappedOperatorAtomicTaskShardBinding> bindings = {
          mk_binding(0, 0, 0),
          mk_binding(1, 0, 1),
          mk_binding(2, 0, 2),
          mk_binding(3, 1, 2),
          mk_binding(4, 1, 3),
      };

      std::optional<nonnegative_int> result =
          mapped_op_task_shard_bindings_are_k_unique_on_slot(
              bindings, TensorSlotName::INPUT);

      std::optional<nonnegative_int> correct = std::nullopt;

      CHECK(result == correct);
    }
  }

  TEST_CASE("mapped_op_task_shard_bindings_are_unique_on_slot") {
    auto ms_coord = [&](int idx) -> MachineSpaceCoordinate {
      return MachineSpaceCoordinate{
          /*node_idx=*/nonnegative_int{idx},
          /*device_idx=*/0_n,
      };
    };

    auto pt_coord = [&](int idx) -> ParallelTensorSpaceCoordinate {
      return ParallelTensorSpaceCoordinate{
          /*sum_component=*/0_n,
          /*discard_copy_component=*/0_n,
          /*shard_components=*/
          FFOrdered<nonnegative_int>{
              0_n,
              nonnegative_int{idx},
          },
      };
    };

    auto mk_binding = [&](int device_idx, int input_idx, int output_idx)
        -> MappedOperatorAtomicTaskShardBinding {
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
          mk_binding(0, 0, 0),
          mk_binding(2, 1, 0),
          mk_binding(1, 2, 0),
          mk_binding(3, 3, 0),
          mk_binding(4, 4, 0),
          mk_binding(6, 5, 0),
      };

      SUBCASE("on the slot in question") {
        bool result = mapped_op_task_shard_bindings_are_unique_on_slot(
            bindings, TensorSlotName::INPUT);

        bool correct = true;

        CHECK(result == correct);
      }

      SUBCASE("but not on the slot in question") {
        bool result = mapped_op_task_shard_bindings_are_unique_on_slot(
            bindings, TensorSlotName::OUTPUT);

        bool correct = false;

        CHECK(result == correct);
      }
    }

    SUBCASE("bindings are not unique on any slot") {
      std::set<MappedOperatorAtomicTaskShardBinding> bindings = {
          mk_binding(0, 0, 0),
          mk_binding(2, 0, 0),
          mk_binding(1, 0, 0),
          mk_binding(3, 0, 0),
          mk_binding(4, 0, 0),
          mk_binding(6, 0, 0),
      };

      bool result = mapped_op_task_shard_bindings_are_unique_on_slot(
          bindings, TensorSlotName::INPUT);

      bool correct = false;

      CHECK(result == correct);
    }
  }

  TEST_CASE("mapped_op_task_shard_binding_project_out_key") {
    auto ms_coord = [&](int idx) -> MachineSpaceCoordinate {
      return MachineSpaceCoordinate{
          /*node_idx=*/nonnegative_int{idx},
          /*device_idx=*/0_n,
      };
    };

    auto pt_coord = [&](int idx) -> ParallelTensorSpaceCoordinate {
      return ParallelTensorSpaceCoordinate{
          /*sum_component=*/0_n,
          /*discard_copy_component=*/0_n,
          /*shard_components=*/
          FFOrdered<nonnegative_int>{
              0_n,
              nonnegative_int{idx},
          },
      };
    };

    MappedOperatorAtomicTaskShardBinding input =
        MappedOperatorAtomicTaskShardBinding{
            /*tensor_coords=*/{
                {TensorSlotName::INPUT, pt_coord(1)},
                {TensorSlotName::WEIGHT, pt_coord(2)},
                {TensorSlotName::OUTPUT, pt_coord(1)},
            },
            /*machine_coord=*/ms_coord(3),
        };

    std::pair<ParallelTensorSpaceCoordinate,
              MappedOperatorAtomicTaskShardBinding>
        result = mapped_op_task_shard_binding_project_out_key(
            input, TensorSlotName::WEIGHT);

    std::pair<ParallelTensorSpaceCoordinate,
              MappedOperatorAtomicTaskShardBinding>
        correct = std::make_pair(pt_coord(2),
                                 MappedOperatorAtomicTaskShardBinding{
                                     /*tensor_coords=*/{
                                         {TensorSlotName::INPUT, pt_coord(1)},
                                         {TensorSlotName::OUTPUT, pt_coord(1)},
                                     },
                                     /*machine_coord=*/ms_coord(3),
                                 });

    CHECK(result == correct);
  }
}
