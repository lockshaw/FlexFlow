#include "op-attrs/parallel_tensor_dim_degrees.h"
#include <doctest/doctest.h>
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "test/utils/doctest/fmt/unordered_map.h"
#include "test/utils/doctest/fmt/unordered_set.h"
#include "test/utils/doctest/fmt/set.h"

using namespace ::FlexFlow; 

static parallel_tensor_dim_idx_t shard_dim_idx_from_raw(int idx) {
  return parallel_tensor_dim_idx_t{ff_dim_t{nonnegative_int{idx}}};
}

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_parallel_tensor_degree_map") {
    ParallelTensorDimDegrees degrees = ParallelTensorDimDegrees{
      SumDegree{3},
      DiscardCopyDegree{1},
      FFOrdered<int>{
        1,
        2,
        1,
      },
    };

    std::unordered_map<parallel_tensor_dim_idx_t, int> result = get_parallel_tensor_degree_map(degrees);
    std::unordered_map<parallel_tensor_dim_idx_t, int> correct = {
      {parallel_tensor_dim_idx_t{ReplicaType::SUM}, 3},
      {parallel_tensor_dim_idx_t{ReplicaType::DISCARD_COPY}, 1},
      {shard_dim_idx_from_raw(0), 1},
      {shard_dim_idx_from_raw(1), 2},
      {shard_dim_idx_from_raw(2), 1},
    };

    CHECK(result == correct);
  }

  TEST_CASE("get_parallel_tensor_space_coordinates") {
    ParallelTensorDimDegrees degrees = ParallelTensorDimDegrees{
      SumDegree{3},
      DiscardCopyDegree{1},
      FFOrdered<int>{
        1,
        2,
        1,
      },
    };

    std::unordered_set<ParallelTensorSpaceCoordinate> result = get_parallel_tensor_space_coordinates(degrees);
    std::unordered_set<ParallelTensorSpaceCoordinate> correct = {
      ParallelTensorSpaceCoordinate{
        /*sum_idx=*/0,
        /*discard_copy_idx=*/0,
        /*shard_idxs=*/FFOrdered<int>{0, 0, 0},
      },
      ParallelTensorSpaceCoordinate{
        /*sum_idx=*/1,
        /*discard_copy_idx=*/0,
        /*shard_idxs=*/FFOrdered<int>{0, 0, 0},
      },
      ParallelTensorSpaceCoordinate{
        /*sum_idx=*/2,
        /*discard_copy_idx=*/0,
        /*shard_idxs=*/FFOrdered<int>{0, 0, 0},
      },
      ParallelTensorSpaceCoordinate{
        /*sum_idx=*/0,
        /*discard_copy_idx=*/0,
        /*shard_idxs=*/FFOrdered<int>{0, 1, 0},
      },
      ParallelTensorSpaceCoordinate{
        /*sum_idx=*/1,
        /*discard_copy_idx=*/0,
        /*shard_idxs=*/FFOrdered<int>{0, 1, 0},
      },
      ParallelTensorSpaceCoordinate{
        /*sum_idx=*/2,
        /*discard_copy_idx=*/0,
        /*shard_idxs=*/FFOrdered<int>{0, 1, 0},
      },
    };

    CHECK(result == correct);
  }

  TEST_CASE("get_nontrivial_parallel_tensor_dim_indices(ParallelTensorDimDegrees)") {
    SUBCASE("a replica dim has degree 1") {
      ParallelTensorDimDegrees degrees = ParallelTensorDimDegrees{
        SumDegree{3},
        DiscardCopyDegree{1},
        FFOrdered<int>{4, 2, 4},
      };

      std::set<parallel_tensor_dim_idx_t> result = get_nontrivial_parallel_tensor_dim_indices(degrees);
      std::set<parallel_tensor_dim_idx_t> correct = {
        parallel_tensor_dim_idx_t{ReplicaType::SUM},
        shard_dim_idx_from_raw(0),
        shard_dim_idx_from_raw(1),
        shard_dim_idx_from_raw(2),
      };

      CHECK(result == correct);
    }

    SUBCASE("a shard dim has degree 1") {
      ParallelTensorDimDegrees degrees = ParallelTensorDimDegrees{
        SumDegree{3},
        DiscardCopyDegree{2},
        FFOrdered<int>{1, 4, 1},
      };

      std::set<parallel_tensor_dim_idx_t> result = get_nontrivial_parallel_tensor_dim_indices(degrees);
      std::set<parallel_tensor_dim_idx_t> correct = {
        parallel_tensor_dim_idx_t{ReplicaType::SUM},
        parallel_tensor_dim_idx_t{ReplicaType::DISCARD_COPY},
        shard_dim_idx_from_raw(1),
      };

      CHECK(result == correct);
    }

    SUBCASE("no dims have degree 1") {
      ParallelTensorDimDegrees degrees = ParallelTensorDimDegrees{
        SumDegree{3},
        DiscardCopyDegree{2},
        FFOrdered<int>{4, 2, 5},
      };

      std::set<parallel_tensor_dim_idx_t> result = get_nontrivial_parallel_tensor_dim_indices(degrees);
      std::set<parallel_tensor_dim_idx_t> correct = {
        parallel_tensor_dim_idx_t{ReplicaType::SUM},
        parallel_tensor_dim_idx_t{ReplicaType::DISCARD_COPY},
        shard_dim_idx_from_raw(0),
        shard_dim_idx_from_raw(1),
        shard_dim_idx_from_raw(2),
      };

      CHECK(result == correct);
    }

    SUBCASE("all dims have degree 1") {
      ParallelTensorDimDegrees degrees = ParallelTensorDimDegrees{
        SumDegree{1},
        DiscardCopyDegree{1},
        FFOrdered<int>{1, 1, 1},
      };

      std::set<parallel_tensor_dim_idx_t> result = get_nontrivial_parallel_tensor_dim_indices(degrees);
      std::set<parallel_tensor_dim_idx_t> correct = {};

      CHECK(result == correct);
    }
  }
}
