#include "op-attrs/parallel_tensor_space_coordinate.h"
#include "test/utils/doctest/fmt/optional.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

static ParallelTensorSpaceCoordinate mk_0d_coord(int sum_component,
                                                 int discard_copy_component) {
  return ParallelTensorSpaceCoordinate{
      /*sum_component=*/nonnegative_int{sum_component},
      /*discard_copy_component=*/nonnegative_int{discard_copy_component},
      /*shard_components=*/FFOrdered<nonnegative_int>{},
  };
}

static ParallelTensorSpaceCoordinate mk_1d_coord(int sum_component,
                                                 int discard_copy_component,
                                                 int shard_0_component) {
  return ParallelTensorSpaceCoordinate{
      /*sum_component=*/nonnegative_int{sum_component},
      /*discard_copy_component=*/nonnegative_int{discard_copy_component},
      /*shard_components=*/
      FFOrdered<nonnegative_int>{
          nonnegative_int{shard_0_component},
      },
  };
}

static ParallelTensorSpaceCoordinate mk_2d_coord(int sum_component,
                                                 int discard_copy_component,
                                                 int shard_0_component,
                                                 int shard_1_component) {
  return ParallelTensorSpaceCoordinate{
      /*sum_component=*/nonnegative_int{sum_component},
      /*discard_copy_component=*/nonnegative_int{discard_copy_component},
      /*shard_components=*/
      FFOrdered<nonnegative_int>{
          nonnegative_int{shard_0_component},
          nonnegative_int{shard_1_component},
      },
  };
}

static ParallelTensorSpaceCoordinate mk_3d_coord(int sum_component,
                                                 int discard_copy_component,
                                                 int shard_0_component,
                                                 int shard_1_component,
                                                 int shard_2_component) {
  return ParallelTensorSpaceCoordinate{
      /*sum_component=*/nonnegative_int{sum_component},
      /*discard_copy_component=*/nonnegative_int{discard_copy_component},
      /*shard_components=*/
      FFOrdered<nonnegative_int>{
          nonnegative_int{shard_0_component},
          nonnegative_int{shard_1_component},
          nonnegative_int{shard_2_component},
      },
  };
}

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("smallest_parallel_tensor_dim_degrees_for_coord_set") {
    SUBCASE("coord set is empty") {
      std::set<ParallelTensorSpaceCoordinate> coord_set = {};

      CHECK_THROWS(
          smallest_parallel_tensor_dim_degrees_for_coord_set(coord_set));
    }

    SUBCASE("coord set contains coords of different dimensionalities") {
      std::set<ParallelTensorSpaceCoordinate> coord_set = {
          mk_1d_coord(1, 2, 0),
          mk_2d_coord(0, 4, 0, 3),
      };

      CHECK_THROWS(
          smallest_parallel_tensor_dim_degrees_for_coord_set(coord_set));
    }

    SUBCASE("coord set is not orthotopic") {
      std::set<ParallelTensorSpaceCoordinate> coord_set = {
          mk_1d_coord(0, 1, 2),
          mk_1d_coord(2, 4, 1),
          mk_1d_coord(3, 0, 0),
      };

      std::optional<ParallelTensorDimDegrees> result =
          smallest_parallel_tensor_dim_degrees_for_coord_set(coord_set);
      std::optional<ParallelTensorDimDegrees> correct =
          ParallelTensorDimDegrees{
              /*sum_degrees=*/SumDegree{4_p},
              /*discard_copy_degrees=*/DiscardCopyDegree{5_p},
              /*shard_degrees=*/
              FFOrdered<positive_int>{
                  3_p,
              },
          };

      CHECK(result == correct);
    }

    SUBCASE("coord set is orthotopic") {
      std::set<ParallelTensorSpaceCoordinate> coord_set = {
          mk_1d_coord(0, 0, 0),
          mk_1d_coord(0, 1, 0),
          mk_1d_coord(0, 2, 0),
          mk_1d_coord(1, 0, 0),
          mk_1d_coord(1, 1, 0),
          mk_1d_coord(1, 2, 0),
          mk_1d_coord(0, 0, 1),
          mk_1d_coord(0, 1, 1),
          mk_1d_coord(0, 2, 1),
          mk_1d_coord(1, 0, 1),
          mk_1d_coord(1, 1, 1),
          mk_1d_coord(1, 2, 1),
      };

      std::optional<ParallelTensorDimDegrees> result =
          smallest_parallel_tensor_dim_degrees_for_coord_set(coord_set);
      std::optional<ParallelTensorDimDegrees> correct =
          ParallelTensorDimDegrees{
              /*sum_degree=*/SumDegree{2_p},
              /*discard_copy_degree=*/DiscardCopyDegree{3_p},
              /*shard_degrees=*/
              FFOrdered<positive_int>{
                  2_p,
              },
          };

      CHECK(result == correct);
    }

    SUBCASE("coord set's space is not minimal") {
      std::set<ParallelTensorSpaceCoordinate> coord_set = {
          mk_1d_coord(0, 0, 0),
          mk_1d_coord(0, 1, 0),
          mk_1d_coord(0, 2, 0),
          mk_1d_coord(1, 0, 0),
          mk_1d_coord(1, 1, 0),
          mk_1d_coord(1, 2, 0),
      };

      std::optional<ParallelTensorDimDegrees> result =
          smallest_parallel_tensor_dim_degrees_for_coord_set(coord_set);
      std::optional<ParallelTensorDimDegrees> correct =
          ParallelTensorDimDegrees{
              /*sum_degree=*/SumDegree{2_p},
              /*discard_copy_degree=*/DiscardCopyDegree{3_p},
              /*shard_degrees=*/
              FFOrdered<positive_int>{
                  1_p,
              },
          };
    }
  }

  TEST_CASE("strict_parallel_tensor_dim_degrees_for_coord_set") {
    SUBCASE("coord set is empty") {
      std::set<ParallelTensorSpaceCoordinate> coord_set = {};

      std::optional<ParallelTensorDimDegrees> result =
          strict_parallel_tensor_dim_degrees_for_coord_set(coord_set);
      std::optional<ParallelTensorDimDegrees> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("coord set contains coords of different dimensionalities") {
      std::set<ParallelTensorSpaceCoordinate> coord_set = {
          mk_1d_coord(1, 2, 0),
          mk_2d_coord(0, 4, 0, 3),
      };

      CHECK_THROWS(strict_parallel_tensor_dim_degrees_for_coord_set(coord_set));
    }

    SUBCASE("coord set is not orthotopic") {
      std::set<ParallelTensorSpaceCoordinate> coord_set = {
          mk_1d_coord(0, 1, 2),
          mk_1d_coord(2, 4, 1),
          mk_1d_coord(3, 0, 0),
      };

      std::optional<ParallelTensorDimDegrees> result =
          strict_parallel_tensor_dim_degrees_for_coord_set(coord_set);
      std::optional<ParallelTensorDimDegrees> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("coord set is orthotopic") {
      std::set<ParallelTensorSpaceCoordinate> coord_set = {
          mk_1d_coord(0, 0, 0),
          mk_1d_coord(0, 1, 0),
          mk_1d_coord(0, 2, 0),
          mk_1d_coord(1, 0, 0),
          mk_1d_coord(1, 1, 0),
          mk_1d_coord(1, 2, 0),
          mk_1d_coord(0, 0, 1),
          mk_1d_coord(0, 1, 1),
          mk_1d_coord(0, 2, 1),
          mk_1d_coord(1, 0, 1),
          mk_1d_coord(1, 1, 1),
          mk_1d_coord(1, 2, 1),
      };

      std::optional<ParallelTensorDimDegrees> result =
          strict_parallel_tensor_dim_degrees_for_coord_set(coord_set);
      std::optional<ParallelTensorDimDegrees> correct =
          ParallelTensorDimDegrees{
              /*sum_degree=*/SumDegree{2_p},
              /*discard_copy_degree=*/DiscardCopyDegree{3_p},
              /*shard_degrees=*/
              FFOrdered<positive_int>{
                  2_p,
              },
          };

      CHECK(result == correct);
    }

    SUBCASE("coord set's space is not minimal") {
      std::set<ParallelTensorSpaceCoordinate> coord_set = {
          mk_1d_coord(0, 0, 0),
          mk_1d_coord(0, 1, 0),
          mk_1d_coord(0, 2, 0),
          mk_1d_coord(1, 0, 0),
          mk_1d_coord(1, 1, 0),
          mk_1d_coord(1, 2, 0),
      };

      std::optional<ParallelTensorDimDegrees> result =
          strict_parallel_tensor_dim_degrees_for_coord_set(coord_set);
      std::optional<ParallelTensorDimDegrees> correct =
          ParallelTensorDimDegrees{
              /*sum_degree=*/SumDegree{2_p},
              /*discard_copy_degree=*/DiscardCopyDegree{3_p},
              /*shard_degrees=*/
              FFOrdered<positive_int>{
                  1_p,
              },
          };
    }
  }

  TEST_CASE("parallel_tensor_coord_set_is_orthotopic") {
    SUBCASE("coord set is empty") {
      std::set<ParallelTensorSpaceCoordinate> coord_set = {};

      bool result = parallel_tensor_coord_set_is_orthotopic(coord_set);
      bool correct = false;

      CHECK(result == correct);
    }

    SUBCASE("coord set contains coords of different dimensionalities") {
      std::set<ParallelTensorSpaceCoordinate> coord_set = {
          mk_1d_coord(1, 2, 0),
          mk_2d_coord(0, 4, 0, 3),
      };

      CHECK_THROWS(parallel_tensor_coord_set_is_orthotopic(coord_set));
    }

    SUBCASE("coord set is not orthotopic") {
      std::set<ParallelTensorSpaceCoordinate> coord_set = {
          mk_1d_coord(0, 1, 2),
          mk_1d_coord(2, 4, 1),
          mk_1d_coord(3, 0, 0),
      };

      bool result = parallel_tensor_coord_set_is_orthotopic(coord_set);
      bool correct = false;

      CHECK(result == correct);
    }

    SUBCASE("coord set is orthotopic") {
      std::set<ParallelTensorSpaceCoordinate> coord_set = {
          mk_1d_coord(0, 0, 0),
          mk_1d_coord(0, 1, 0),
          mk_1d_coord(0, 2, 0),
          mk_1d_coord(1, 0, 0),
          mk_1d_coord(1, 1, 0),
          mk_1d_coord(1, 2, 0),
          mk_1d_coord(0, 0, 1),
          mk_1d_coord(0, 1, 1),
          mk_1d_coord(0, 2, 1),
          mk_1d_coord(1, 0, 1),
          mk_1d_coord(1, 1, 1),
          mk_1d_coord(1, 2, 1),
      };

      bool result = parallel_tensor_coord_set_is_orthotopic(coord_set);
      bool correct = true;

      CHECK(result == correct);
    }

    SUBCASE("coord set's space is orthotopic but not minimal") {
      std::set<ParallelTensorSpaceCoordinate> coord_set = {
          mk_1d_coord(0, 0, 0),
          mk_1d_coord(0, 1, 0),
          mk_1d_coord(0, 2, 0),
          mk_1d_coord(1, 0, 0),
          mk_1d_coord(1, 1, 0),
          mk_1d_coord(1, 2, 0),
      };

      bool result = parallel_tensor_coord_set_is_orthotopic(coord_set);
      bool correct = true;

      CHECK(result == correct);
    }
  }
}
