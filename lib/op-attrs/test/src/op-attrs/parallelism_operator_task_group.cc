#include "op-attrs/parallelism_operator_task_group.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

static ParallelTensorSpaceCoordinate make_1d_pt_coord(
    int sum_component, int discard_copy_component, int dim0_component) {
  return ParallelTensorSpaceCoordinate{
      /*sum_component=*/nonnegative_int{sum_component},
      /*discard_copy_component=*/nonnegative_int{discard_copy_component},
      /*shard_components=*/
      FFOrdered<nonnegative_int>{
          nonnegative_int{dim0_component},
      },
  };
}

static ParallelTensorSpaceCoordinate
    make_2d_pt_coord(int sum_component,
                     int discard_copy_component,
                     int dim0_component,
                     int dim1_component) {
  return ParallelTensorSpaceCoordinate{
      /*sum_component=*/nonnegative_int{sum_component},
      /*discard_copy_component=*/nonnegative_int{discard_copy_component},
      /*shard_components=*/
      FFOrdered<nonnegative_int>{
          nonnegative_int{dim0_component},
          nonnegative_int{dim1_component},
      },
  };
}

static ParallelTensorSpaceCoordinate
    make_3d_pt_coord(int sum_component,
                     int discard_copy_component,
                     int dim0_component,
                     int dim1_component,
                     int dim2_component) {
  return ParallelTensorSpaceCoordinate{
      /*sum_component=*/nonnegative_int{sum_component},
      /*discard_copy_component=*/nonnegative_int{discard_copy_component},
      /*shard_components=*/
      FFOrdered<nonnegative_int>{
          nonnegative_int{dim0_component},
          nonnegative_int{dim1_component},
          nonnegative_int{dim2_component},
      },
  };
}

static TaskSpaceCoordinate make_1d_task_coord(int dim0_component) {
  return TaskSpaceCoordinate{
      OrthotopeCoord{
          std::vector<nonnegative_int>{
              nonnegative_int{dim0_component},
          },
      },
  };
}

static TaskSpaceCoordinate make_2d_task_coord(int dim0_component,
                                              int dim1_component) {
  return TaskSpaceCoordinate{
      OrthotopeCoord{
          std::vector<nonnegative_int>{
              nonnegative_int{dim0_component},
              nonnegative_int{dim1_component},
          },
      },
  };
}

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("ParallelismOperatorTaskGroup constructor") {

    auto one_slot_binding =
        [&](ParallelTensorSpaceCoordinate const &output_coord,
            TaskSpaceCoordinate const &task_space_coord)
        -> AbstractedOperatorAtomicTaskShardBinding {
      return AbstractedOperatorAtomicTaskShardBinding{
          /*tensor_coords=*/std::map<TensorSlotName,
                                     ParallelTensorSpaceCoordinate>{
              {
                  TensorSlotName::OUTPUT,
                  output_coord,
              },
          },
          /*task_coord=*/task_space_coord,
      };
    };

    auto two_slot_binding =
        [&](ParallelTensorSpaceCoordinate const &input_coord,
            ParallelTensorSpaceCoordinate const &output_coord,
            TaskSpaceCoordinate const &task_space_coord)
        -> AbstractedOperatorAtomicTaskShardBinding {
      return AbstractedOperatorAtomicTaskShardBinding{
          /*tensor_coords=*/std::map<TensorSlotName,
                                     ParallelTensorSpaceCoordinate>{
              {
                  TensorSlotName::INPUT,
                  input_coord,
              },
              {
                  TensorSlotName::OUTPUT,
                  output_coord,
              },
          },
          /*task_coord=*/task_space_coord,
      };
    };

    auto three_slot_binding =
        [&](ParallelTensorSpaceCoordinate const &input_coord,
            ParallelTensorSpaceCoordinate const &weight_coord,
            ParallelTensorSpaceCoordinate const &output_coord,
            TaskSpaceCoordinate const &task_space_coord)
        -> AbstractedOperatorAtomicTaskShardBinding {
      return AbstractedOperatorAtomicTaskShardBinding{
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
          /*task_coord=*/task_space_coord,
      };
    };

    SUBCASE("has more slots than INPUT and OUTPUT") {
      std::set<AbstractedOperatorAtomicTaskShardBinding> input = {
          three_slot_binding(make_1d_pt_coord(0, 0, 0),
                             make_1d_pt_coord(0, 0, 0),
                             make_1d_pt_coord(0, 0, 0),
                             make_1d_task_coord(0)),
          three_slot_binding(make_1d_pt_coord(0, 0, 0),
                             make_1d_pt_coord(0, 0, 1),
                             make_1d_pt_coord(0, 0, 1),
                             make_1d_task_coord(1)),
      };

      CHECK_THROWS(ParallelismOperatorTaskGroup{input});
    }

    SUBCASE("has fewer slots than INPUT and OUTPUT") {
      std::set<AbstractedOperatorAtomicTaskShardBinding> input = {
          one_slot_binding(make_1d_pt_coord(0, 0, 0), make_1d_task_coord(0)),
          one_slot_binding(make_1d_pt_coord(0, 0, 1), make_1d_task_coord(1)),
      };

      CHECK_THROWS(ParallelismOperatorTaskGroup{input});
    }

    SUBCASE("input ptensor space is not orthotopic") {
      std::set<AbstractedOperatorAtomicTaskShardBinding> input = {
          two_slot_binding(make_1d_pt_coord(0, 0, 0),
                           make_1d_pt_coord(0, 0, 0),
                           make_1d_task_coord(0)),
          two_slot_binding(make_1d_pt_coord(0, 0, 1),
                           make_1d_pt_coord(0, 0, 0),
                           make_1d_task_coord(1)),
          two_slot_binding(make_1d_pt_coord(0, 1, 0),
                           make_1d_pt_coord(0, 1, 0),
                           make_1d_task_coord(2)),
      };

      CHECK_THROWS(ParallelismOperatorTaskGroup{input});
    }

    SUBCASE("output ptensor space is not orthotopic") {
      std::set<AbstractedOperatorAtomicTaskShardBinding> input = {
          two_slot_binding(make_1d_pt_coord(0, 0, 0),
                           make_1d_pt_coord(0, 0, 0),
                           make_1d_task_coord(0)),
          two_slot_binding(make_1d_pt_coord(0, 0, 0),
                           make_1d_pt_coord(0, 0, 1),
                           make_1d_task_coord(1)),
          two_slot_binding(make_1d_pt_coord(0, 1, 0),
                           make_1d_pt_coord(0, 1, 0),
                           make_1d_task_coord(2)),
      };

      CHECK_THROWS(ParallelismOperatorTaskGroup{input});
    }

    SUBCASE("task coord space is not orthotopic") {
      std::set<AbstractedOperatorAtomicTaskShardBinding> input = {
          two_slot_binding(make_1d_pt_coord(0, 0, 0),
                           make_1d_pt_coord(0, 0, 0),
                           make_2d_task_coord(0, 0)),
          two_slot_binding(make_1d_pt_coord(0, 0, 0),
                           make_1d_pt_coord(0, 0, 1),
                           make_2d_task_coord(0, 1)),
          two_slot_binding(make_1d_pt_coord(0, 0, 0),
                           make_1d_pt_coord(0, 0, 2),
                           make_2d_task_coord(1, 0)),
      };

      CHECK_THROWS(ParallelismOperatorTaskGroup{input});
    }

    SUBCASE("coord relation is bi-unique") {
      std::set<AbstractedOperatorAtomicTaskShardBinding> input = {
          two_slot_binding(make_1d_pt_coord(0, 0, 0),
                           make_1d_pt_coord(0, 0, 0),
                           make_1d_task_coord(0)),
          two_slot_binding(make_1d_pt_coord(0, 0, 1),
                           make_1d_pt_coord(0, 0, 1),
                           make_1d_task_coord(1)),
      };

      CHECK_THROWS(ParallelismOperatorTaskGroup{input});
    }

    SUBCASE("coord relation scales two dimensions at a time") {
      std::set<AbstractedOperatorAtomicTaskShardBinding> input = {
          two_slot_binding(make_1d_pt_coord(0, 0, 0),
                           make_1d_pt_coord(0, 0, 0),
                           make_2d_task_coord(0, 0)),
          two_slot_binding(make_1d_pt_coord(0, 0, 0),
                           make_1d_pt_coord(0, 0, 1),
                           make_2d_task_coord(0, 1)),
          two_slot_binding(make_1d_pt_coord(0, 0, 0),
                           make_1d_pt_coord(0, 1, 0),
                           make_2d_task_coord(1, 0)),
          two_slot_binding(make_1d_pt_coord(0, 0, 0),
                           make_1d_pt_coord(0, 1, 1),
                           make_2d_task_coord(1, 1)),
      };

      CHECK_THROWS(ParallelismOperatorTaskGroup{input});
    }

    SUBCASE("coord relation is like Repartition") {
      std::set<AbstractedOperatorAtomicTaskShardBinding> input = {
          two_slot_binding(make_1d_pt_coord(0, 0, 0),
                           make_1d_pt_coord(0, 0, 0),
                           make_2d_task_coord(0, 0)),
          two_slot_binding(make_1d_pt_coord(0, 0, 0),
                           make_1d_pt_coord(0, 0, 1),
                           make_2d_task_coord(0, 1)),
          two_slot_binding(make_1d_pt_coord(0, 1, 0),
                           make_1d_pt_coord(0, 1, 0),
                           make_2d_task_coord(1, 0)),
          two_slot_binding(make_1d_pt_coord(0, 1, 0),
                           make_1d_pt_coord(0, 1, 1),
                           make_2d_task_coord(1, 1)),
      };

      // check does not throw
      ParallelismOperatorTaskGroup{input};
    }

    SUBCASE("coord relation is like Combine") {
      std::set<AbstractedOperatorAtomicTaskShardBinding> input = {
          two_slot_binding(make_1d_pt_coord(0, 0, 0),
                           make_1d_pt_coord(0, 0, 0),
                           make_2d_task_coord(0, 0)),
          two_slot_binding(make_1d_pt_coord(0, 0, 1),
                           make_1d_pt_coord(0, 0, 0),
                           make_2d_task_coord(0, 1)),
          two_slot_binding(make_1d_pt_coord(0, 1, 0),
                           make_1d_pt_coord(0, 1, 0),
                           make_2d_task_coord(1, 0)),
          two_slot_binding(make_1d_pt_coord(0, 1, 1),
                           make_1d_pt_coord(0, 1, 0),
                           make_2d_task_coord(1, 1)),
      };

      // check does not throw
      ParallelismOperatorTaskGroup{input};
    }

    SUBCASE("coord relation is like Replicate") {
      std::set<AbstractedOperatorAtomicTaskShardBinding> input = {
          two_slot_binding(make_1d_pt_coord(0, 0, 0),
                           make_1d_pt_coord(0, 0, 0),
                           make_2d_task_coord(0, 0)),
          two_slot_binding(make_1d_pt_coord(0, 0, 0),
                           make_1d_pt_coord(0, 1, 0),
                           make_2d_task_coord(0, 1)),
          two_slot_binding(make_1d_pt_coord(0, 0, 1),
                           make_1d_pt_coord(0, 0, 1),
                           make_2d_task_coord(1, 0)),
          two_slot_binding(make_1d_pt_coord(0, 0, 1),
                           make_1d_pt_coord(0, 1, 1),
                           make_2d_task_coord(1, 1)),
      };

      // check does not throw
      ParallelismOperatorTaskGroup{input};
    }

    SUBCASE("coord relation is like Reduction") {
      std::set<AbstractedOperatorAtomicTaskShardBinding> input = {
          two_slot_binding(make_1d_pt_coord(0, 0, 0),
                           make_1d_pt_coord(0, 0, 0),
                           make_2d_task_coord(0, 0)),
          two_slot_binding(make_1d_pt_coord(1, 0, 0),
                           make_1d_pt_coord(0, 0, 0),
                           make_2d_task_coord(0, 1)),
          two_slot_binding(make_1d_pt_coord(0, 0, 1),
                           make_1d_pt_coord(0, 0, 1),
                           make_2d_task_coord(1, 0)),
          two_slot_binding(make_1d_pt_coord(1, 0, 1),
                           make_1d_pt_coord(0, 0, 1),
                           make_2d_task_coord(1, 1)),
      };

      // check does not throw
      ParallelismOperatorTaskGroup{input};
    }
  }
}
