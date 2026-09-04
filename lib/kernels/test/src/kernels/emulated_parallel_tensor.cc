#include <doctest/doctest.h>
#include "kernels/emulated_parallel_tensor.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("emulated_parallel_tensors_are_equal") {
    auto mk_coord = [](int sum_component,
                       int discard_copy_component,
                       int shard_dim0_component,
                       int shard_dim1_component)
      -> ParallelTensorSpaceCoordinate
    {
      return ParallelTensorSpaceCoordinate{
        /*sum_component=*/nonnegative_int{sum_component},
        /*discard_copy_component=*/nonnegative_int{discard_copy_component},
        /*shard_components=*/FFOrdered<nonnegative_int>{
          nonnegative_int{shard_dim0_component},
          nonnegative_int{shard_dim1_component},
        },
      };
    };

    GenericTensorAccessorR input_shard1 = create_2d_accessor_r_with_contents<float>(
        {
            {3, 3, 6},
            {2, 1, 5},
        },
        cpu_allocator);

    GenericTensorAccessorR input_shard2 = create_2d_accessor_r_with_contents<float>(
        {
            {5, 1, 2},
            {3, 3, 7},
        },
        cpu_allocator);

    GenericTensorAccessorR input_shard3 = create_2d_accessor_r_with_contents<float>(
        {
            {4, 0, 9},
            {1, 2, -3},
        },
        cpu_allocator);

    GenericTensorAccessorR input_shard4 = create_2d_accessor_r_with_contents<float>(
        {
            {-2, 3, 4},
            {5, 1, 0},
        },
        cpu_allocator);

    EmulatedParallelTensor input1 = EmulatedParallelTensor{
      /*shards=*/std::map<ParallelTensorSpaceCoordinate, GenericTensorAccessorR>{
        {
          mk_coord(0, 0, 0, 0),
          input_shard1,
        },
        {
          mk_coord(1, 0, 0, 0),
          input_shard2,
        },
        {
          mk_coord(0, 0, 1, 0),
          input_shard3,
        },
        {
          mk_coord(1, 0, 1, 0),
          input_shard4,
        },
      },
    };

    SUBCASE("are equal") {
      CHECK(emulated_parallel_tensors_are_equal(input1, input1));
    }

    SUBCASE("have different coordinate sets") {
      EmulatedParallelTensor input2 = EmulatedParallelTensor{
        /*shards=*/std::map<ParallelTensorSpaceCoordinate, GenericTensorAccessorR>{
          {
            mk_coord(0, 0, 0, 0),
            input_shard1,
          },
          {
            mk_coord(0, 0, 1, 0),
            input_shard2,
          },
          {
            mk_coord(0, 0, 2, 0),
            input_shard3,
          },
          {
            mk_coord(0, 0, 3, 0),
            input_shard4,
          },
        },
      };

      CHECK_FALSE(emulated_parallel_tensors_are_equal(input1, input2));
      CHECK_FALSE(emulated_parallel_tensors_are_equal(input2, input1));
    }

    SUBCASE("have different tensor values") {
      EmulatedParallelTensor input2 = EmulatedParallelTensor{
        /*shards=*/std::map<ParallelTensorSpaceCoordinate, GenericTensorAccessorR>{
          {
            mk_coord(0, 0, 0, 0),
            input_shard2,
          },
          {
            mk_coord(1, 0, 0, 0),
            input_shard1,
          },
          {
            mk_coord(0, 0, 1, 0),
            input_shard3,
          },
          {
            mk_coord(1, 0, 1, 0),
            input_shard4,
          },
        },
      };

      CHECK_FALSE(emulated_parallel_tensors_are_equal(input1, input2));
      CHECK_FALSE(emulated_parallel_tensors_are_equal(input2, input1));
    }

    SUBCASE("is subset") {
      EmulatedParallelTensor input2 = EmulatedParallelTensor{
        /*shards=*/std::map<ParallelTensorSpaceCoordinate, GenericTensorAccessorR>{
          {
            mk_coord(0, 0, 0, 0),
            input_shard1,
          },
          {
            mk_coord(1, 0, 0, 0),
            input_shard2,
          },
        },
      };

      CHECK_FALSE(emulated_parallel_tensors_are_equal(input1, input2));
      CHECK_FALSE(emulated_parallel_tensors_are_equal(input2, input1));
    }
  }
}
