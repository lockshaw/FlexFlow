#include "op-attrs/abstracted_operator_atomic_task_shard_binding.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("restrict_abstracted_operator_atomic_task_shard_binding_to_slots") {
    ParallelTensorSpaceCoordinate input_coord = ParallelTensorSpaceCoordinate{
        /*sum_component=*/0_n,
        /*discard_copy_component=*/3_n,
        /*shard_components=*/
        FFOrdered<nonnegative_int>{
            2_n,
            4_n,
        },
    };

    ParallelTensorSpaceCoordinate weight_coord = ParallelTensorSpaceCoordinate{
        /*sum_component=*/1_n,
        /*discard_copy_component=*/1_n,
        /*shard_components=*/
        FFOrdered<nonnegative_int>{
            8_n,
        },
    };

    ParallelTensorSpaceCoordinate output_coord = ParallelTensorSpaceCoordinate{
        /*sum_component=*/0_n,
        /*discard_copy_component=*/0_n,
        /*shard_components=*/
        FFOrdered<nonnegative_int>{
            6_n,
            5_n,
            3_n,
        },
    };

    TaskSpaceCoordinate task_coord = TaskSpaceCoordinate{
        OrthotopeCoord{
            {2_n, 0_n, 5_n},
        },
    };

    AbstractedOperatorAtomicTaskShardBinding binding =
        AbstractedOperatorAtomicTaskShardBinding{
            /*tensor_coords=*/std::map<TensorSlotName,
                                       ParallelTensorSpaceCoordinate>{
                {
                    TensorSlotName::INPUT,
                    input_coord,
                },
                {
                    TensorSlotName::WEIGHT,
                    weight_coord,
                },
                {
                    TensorSlotName::OUTPUT,
                    output_coord,
                },
            },
            /*task_coord=*/task_coord,
        };

    SUBCASE("desired slots is subset of binding slots") {
      std::set<TensorSlotName> desired_slots = {
          TensorSlotName::WEIGHT,
          TensorSlotName::OUTPUT,
      };

      AbstractedOperatorAtomicTaskShardBinding result =
          restrict_abstracted_operator_atomic_task_shard_binding_to_slots(
              binding, desired_slots);

      AbstractedOperatorAtomicTaskShardBinding correct =
          AbstractedOperatorAtomicTaskShardBinding{
              /*tensor_coords=*/std::map<TensorSlotName,
                                         ParallelTensorSpaceCoordinate>{
                  {
                      TensorSlotName::WEIGHT,
                      weight_coord,
                  },
                  {
                      TensorSlotName::OUTPUT,
                      output_coord,
                  },
              },
              /*task_coord=*/task_coord,
          };

      CHECK(result == correct);
    }

    SUBCASE("desired slots includes slots outside of binding slots") {
      std::set<TensorSlotName> desired_slots = {
          TensorSlotName::WEIGHT,
          TensorSlotName::GAMMA,
      };

      CHECK_THROWS(
          restrict_abstracted_operator_atomic_task_shard_binding_to_slots(
              binding, desired_slots));
    }
  }
}
