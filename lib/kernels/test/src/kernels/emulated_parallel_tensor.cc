#include <doctest/doctest.h>
#include "kernels/emulated_parallel_tensor.h"
#include "kernels/create_accessor_with_contents.h"
#include "kernels/accessors_are_equal.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("random_emulated_parallel_tensor_of_shape") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    SUBCASE("is properly deterministic") {
      ParallelTensorShape shape = ParallelTensorShape{
        /*dims=*/ParallelTensorDims{
          /*shard_dims=*/FFOrdered{
            ShardParallelDim{4_p, 2_p},
            ShardParallelDim{3_p, 1_p},
            ShardParallelDim{9_p, 3_p},
          },
          /*replica_dims=*/ReplicaParallelDimSet{
            SumDegree{1_p},
            DiscardCopyDegree{1_p},
          },
        },
        /*data_type=*/DataType::FLOAT,
      };

      EmulatedParallelTensor t1 = random_emulated_parallel_tensor_of_shape(shape, cpu_allocator, 1);

      SUBCASE("tensors with the same seed are equal") {
        EmulatedParallelTensor t2 = random_emulated_parallel_tensor_of_shape(shape, cpu_allocator, 1);

        CHECK(emulated_parallel_tensors_are_equal(t1, t2));
      }

      SUBCASE("tensors with different seeds are not equal") {
        EmulatedParallelTensor t2 = random_emulated_parallel_tensor_of_shape(shape, cpu_allocator, 2);

        CHECK_FALSE(emulated_parallel_tensors_are_equal(t1, t2));
      }
    }

    SUBCASE("tensors in the discard copy dimension are all the same") {
      ParallelTensorShape shape = ParallelTensorShape{
        /*dims=*/ParallelTensorDims{
          /*shard_dims=*/FFOrdered{
            ShardParallelDim{4_p, 1_p},
            ShardParallelDim{6_p, 2_p},
          },
          /*replica_dims=*/ReplicaParallelDimSet{
            SumDegree{1_p},
            DiscardCopyDegree{3_p},
          },
        },
        /*data_type=*/DataType::FLOAT,
      };

      EmulatedParallelTensor t = random_emulated_parallel_tensor_of_shape(shape, cpu_allocator, 1);

      auto shards_are_equal = [&](ParallelTensorSpaceCoordinate const &c1,
                                  ParallelTensorSpaceCoordinate const &c2) 
        -> bool
      {
        return accessors_are_equal(t.shards.at(c1), t.shards.at(c2));
      };

      auto coord = [](int discard_copy_component, int shard_component) -> ParallelTensorSpaceCoordinate {
        return ParallelTensorSpaceCoordinate{
          /*sum_component=*/0_n,
          /*discard_copy_component=*/nonnegative_int{discard_copy_component},
          /*shard_components=*/FFOrdered{
            0_n,
            nonnegative_int{shard_component},
          },
        };
      };

      CHECK(shards_are_equal(coord(0, 0), coord(1, 0)));
      CHECK(shards_are_equal(coord(0, 0), coord(2, 0)));

      CHECK(shards_are_equal(coord(0, 1), coord(1, 1)));
      CHECK(shards_are_equal(coord(0, 1), coord(2, 1)));

      CHECK_FALSE(shards_are_equal(coord(0, 0), coord(0, 1)));
    }
  }

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

    Allocator cpu_allocator = create_local_cpu_memory_allocator();

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
