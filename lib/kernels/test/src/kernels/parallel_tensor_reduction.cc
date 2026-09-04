#include <doctest/doctest.h>
#include "kernels/parallel_tensor_reduction.h"

using namespace ::FlexFlow;

GenericTensorAccessorR mk_shard1(Allocator &allocator) {
    return create_2d_accessor_r_with_contents<float>(
        {
            {3, 3, 6},
            {2, 1, 5},
        },
        allocator);
}

GenericTensorAccessorR mk_shard2(Allocator &allocator) {
    return create_2d_accessor_r_with_contents<float>(
        {
            {5, 1, 2},
            {3, 3, 7},
        },
        allocator);
}

GenericTensorAccessorR mk_shard3(Allocator &allocator) {
    return create_2d_accessor_r_with_contents<float>(
        {
            {4, 0, 9},
            {1, 2, -3},
        },
        allocator);
}

GenericTensorAccessorR mk_shard4(Allocator &allocator) {
    return create_2d_accessor_r_with_contents<float>(
        {
            {-2, 3, 4},
            {5, 1, 0},
        },
        allocator);

}

ParallelTensorSpaceCoordinate mk_coord(
  int sum_component
  int discard_copy_component,
  int shard_dim0_component,
  int shard_dim1_component)
{
  return ParallelTensorSpaceCoordinate{
    /*sum_component=*/nonnegative_int{sum_component},
    /*discard_copy_component=*/nonnegative_int{discard_copy_component},
    /*shard_components=*/FFOrdered<nonnegative_int>{
      nonnegative_int{shard_dim0_component},
      nonnegative_int{shard_dim1_component},
    },
  };
}

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("perform_parallel_tensor_reduction") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    EmulatedParallelTensor input = EmulatedParallelTensor{
      /*shards=*/std::map<ParallelTensorSpaceCoordinate, GenericTensorAccessorR>{
        {
          mk_coord(0, 0, 0, 0),
          mk_shard1(allocator),
        },
        {
          mk_coord(1, 0, 0, 0),
          mk_shard2(allocator),
        },
        {
          mk_coord(0, 0, 1, 0),
          mk_shard3(allocator),
        },
        {
          mk_coord(1, 0, 1, 0),
          mk_shard4(allocator),
        },
      },
    };

    EmulatedParallelTensor result = perform_parallel_tensor_reduction(input, cpu_allocator);

    GenericTensorAccessorR correct_shard1 = create_2d_accessor_r_with_contents<float>(
        {
            {8, 4, 8},
            {5, 4, 12},
        },
        cpu_allocator);

    GenericTensorAccessorR correct_shard2 = create_2d_accessor_r_with_contents<float>(
        {
            {2, 3, 13},
            {6, 3, -3},
        },
        cpu_allocator);

    EmulatedParallelTensor correct = EmulatedParallelTensor{
      /*shards=*/std::map<ParallelTensorSpaceCoordinate, GenericTensorAccessorR>{
        {
          mk_coord(0, 0, 0, 0),
          correct_shard1,
        },
        {
          mk_coord(0, 0, 1, 0),
          correct_shard2,
        },
      },
    };

    CHECK(emulated_parallel_tensors_are_equal(result, correct));
  }

  TEST_CASE("perform_parallel_tensor_discard_copy") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR shard1 = mk_shard1(allocator);
    GenericTensorAccessorR shard2 = mk_shard2(allocator);
    GenericTensorAccessorR shard3 = mk_shard3(allocator);
    GenericTensorAccessorR shard4 = mk_shard4(allocator);

    SUBCASE("are actually copies") {
      EmulatedParallelTensor input = EmulatedParallelTensor{
        /*shards=*/std::map<ParallelTensorSpaceCoordinate, GenericTensorAccessorR>{
          {
            mk_coord(0, 0, 0, 0),
            shard1,
          },
          {
            mk_coord(0, 1, 0, 0),
            shard1,
          },
          {
            mk_coord(0, 0, 1, 0),
            shard2,
          },
          {
            mk_coord(0, 1, 1, 0),
            shard2,
          },
        },
      };

      EmulatedParallelTensor result = perform_parallel_tensor_discard_copy(input);

      EmulatedParallelTensor correct = EmulatedParallelTensor{
        /*shards=*/std::map<ParallelTensorSpaceCoordinate, GenericTensorAccessorR>{
          {
            mk_coord(0, 0, 0, 0),
            shard1,
          },
          {
            mk_coord(0, 0, 1, 0),
            shard2,
          },
        },
      };

      CHECK(emulated_parallel_tensors_are_equal(result, correct));
    }

    SUBCASE("are not actually copies") {
      EmulatedParallelTensor input = EmulatedParallelTensor{
        /*shards=*/std::map<ParallelTensorSpaceCoordinate, GenericTensorAccessorR>{
          {
            mk_coord(0, 0, 0, 0),
            shard1,
          },
          {
            mk_coord(0, 1, 0, 0),
            shard3,
          },
          {
            mk_coord(0, 0, 1, 0),
            shard2,
          },
          {
            mk_coord(0, 1, 1, 0),
            shard2,
          },
        },
      };

      CHECK_THROWS(perform_parallel_tensor_discard_copy(input));
    }
  }

  TEST_CASE("perform_parallel_tensor_combination") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    EmulatedParallelTensor input = EmulatedParallelTensor{
      /*shards=*/std::map<ParallelTensorSpaceCoordinate, GenericTensorAccessorR>{
        {
          mk_coord(0, 0, 0, 0),
          mk_shard1(allocator),
        },
        {
          mk_coord(0, 0, 0, 1),
          mk_shard2(allocator),
        },
        {
          mk_coord(0, 0, 1, 0),
          mk_shard3(allocator),
        },
        {
          mk_coord(0, 0, 1, 1),
          mk_shard4(allocator),
        },
      },
    };

    SUBCASE("combine in dim 1") {
      EmulatedParallelTensor result = perform_parallel_tensor_combination(input, ff_dim_t{1_n}, cpu_allocator);

      GenericTensorAccessorR correct_shard1 = create_2d_accessor_r_with_contents<float>(
          {
              {3, 3, 6, 5, 1, 2},
              {2, 1, 5, 3, 3, 7},
          },
          cpu_allocator);

      GenericTensorAccessorR correct_shard2 = create_2d_accessor_r_with_contents<float>(
          {
              {4, 0, 9, -2, 3, 4},
              {1, 2, -3, 5, 1, 0},
          },
          cpu_allocator);

      EmulatedParallelTensor correct = EmulatedParallelTensor{
        /*shards=*/std::map<ParallelTensorSpaceCoordinate, GenericTensorAccessorR>{
          {
            mk_coord(0, 0, 0, 0),
            correct_shard1,
          },
          {
            mk_coord(0, 0, 1, 0),
            correct_shard2,
          },
        },
      };

      CHECK(emulated_parallel_tensors_are_equal(result, correct));
    }

    SUBCASE("combine in dim 0") {
      EmulatedParallelTensor result = perform_parallel_tensor_combination(input, ff_dim_t{0_n}, cpu_allocator);

      GenericTensorAccessorR correct_shard1 = create_2d_accessor_r_with_contents<float>(
          {
              {3, 3, 6},
              {2, 1, 5},
              {4, 0, 9},
              {1, 2, -3},
          },
          cpu_allocator);

      GenericTensorAccessorR correct_shard2 = create_2d_accessor_r_with_contents<float>(
          {
              {5, 1, 2},
              {3, 3, 7},
              {-2, 3, 4},
              {5, 1, 0},
          },
          cpu_allocator);

      EmulatedParallelTensor correct = EmulatedParallelTensor{
        /*shards=*/std::map<ParallelTensorSpaceCoordinate, GenericTensorAccessorR>{
          {
            mk_coord(0, 0, 0, 0),
            correct_shard1,
          },
          {
            mk_coord(0, 0, 0, 1),
            correct_shard2,
          },
        },
      };

      CHECK(emulated_parallel_tensors_are_equal(result, correct));
    }
  }

  TEST_CASE("unparallelize_parallel_tensor") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    EmulatedParallelTensor input = EmulatedParallelTensor{
      /*shards=*/std::map<ParallelTensorSpaceCoordinate, GenericTensorAccessorR>{
        {
          mk_coord(0, 0, 0, 0),
          mk_shard1(allocator),
        },
        {
          mk_coord(0, 0, 0, 1),
          mk_shard2(allocator),
        },
        {
          mk_coord(0, 1, 0, 0),
          mk_shard1(allocator),
        },
        {
          mk_coord(0, 1, 0, 1),
          mk_shard2(allocator),
        },
        {
          mk_coord(1, 0, 0, 0),
          mk_shard3(allocator),
        },
        {
          mk_coord(1, 0, 0, 1),
          mk_shard4(allocator),
        },
        {
          mk_coord(1, 1, 0, 0),
          mk_shard3(allocator),
        },
        {
          mk_coord(1, 1, 0, 1),
          mk_shard4(allocator),
        },
      },
    };

    GenericTensorAccessorR result = unparallelize_parallel_tensor(input, cpu_allocator);

    GenericTensorAccessorR correct = create_2d_accessor_r_with_contents<float>(
        {
            {7, 0, 15, 3, 4, 6},
            {3, 3, 2, 8, 4, 7},
        },
        cpu_allocator);

    CHECK_MESSAGE(
        accessors_are_equal(result, correct),
        check_kv("result", format_accessor_w_contents(result)));
  }
}
