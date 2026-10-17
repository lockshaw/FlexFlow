#include "op-attrs/ops/reshape.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("reshape_get_output_shape") {
    ReshapeAttrs attrs = ReshapeAttrs{
        TensorDims{FFOrdered<positive_int>{6_p, 5_p}},
        TensorDims{FFOrdered<positive_int>{3_p, 10_p}},
    };

    SUBCASE("input tensor has different num elements") {
      TensorShape input = TensorShape{
          TensorDims{FFOrdered<positive_int>{4_p, 5_p, 5_p}},
          DataType::FLOAT,
      };

      CHECK_THROWS(reshape_get_output_shape(attrs, input));
    }

    SUBCASE("input tensor has different datatype") {
      TensorShape input = TensorShape{
          TensorDims{FFOrdered<positive_int>{4_p, 6_p, 5_p}},
          DataType::DOUBLE,
      };

      TensorShape result = reshape_get_output_shape(attrs, input);
      TensorShape correct = TensorShape{
          TensorDims{FFOrdered<positive_int>{4_p, 3_p, 10_p}},
          DataType::DOUBLE,
      };

      CHECK(result == correct);
    }

    SUBCASE("valid input") {
      TensorShape input = TensorShape{
          TensorDims{FFOrdered<positive_int>{4_p, 6_p, 5_p}},
          DataType::FLOAT,
      };

      TensorShape result = reshape_get_output_shape(attrs, input);

      TensorShape correct = TensorShape{
          TensorDims{FFOrdered<positive_int>{4_p, 3_p, 10_p}},
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }

    SUBCASE("input tensor exactly matches core input dims") {
      TensorShape input = TensorShape{
          TensorDims{FFOrdered<positive_int>{6_p, 5_p}},
          DataType::FLOAT,
      };

      TensorShape result = reshape_get_output_shape(attrs, input);

      TensorShape correct = TensorShape{
          TensorDims{FFOrdered<positive_int>{3_p, 10_p}},
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }

    SUBCASE("input tensor has fewer dims than core input dims") {
      TensorShape input = TensorShape{
          TensorDims{FFOrdered<positive_int>{5_p}},
          DataType::FLOAT,
      };

      CHECK_THROWS(reshape_get_output_shape(attrs, input));
    }
  }

  TEST_CASE("reshape_get_output_parallel_dim_degrees") {

    SUBCASE("preserves number of dims") {
      ReshapeAttrs attrs = ReshapeAttrs{
          TensorDims{FFOrdered<positive_int>{12_p, 5_p}},
          TensorDims{FFOrdered<positive_int>{6_p, 10_p}},
      };

      SUBCASE("input sum degree > 1") {
        ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
            SumDegree{3_p},
            DiscardCopyDegree{1_p},
            FFOrdered<positive_int>{1_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            reshape_get_output_parallel_dim_degrees(attrs, input);
        ParallelTensorDimDegrees correct = input;

        CHECK(result == correct);
      }

      SUBCASE("input discard copy degree > 1") {
        ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
            SumDegree{1_p},
            DiscardCopyDegree{2_p},
            FFOrdered<positive_int>{1_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            reshape_get_output_parallel_dim_degrees(attrs, input);
        ParallelTensorDimDegrees correct = input;

        CHECK(result == correct);
      }

      SUBCASE("allows leading shard degree") {
        ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
            SumDegree{1_p},
            DiscardCopyDegree{1_p},
            FFOrdered<positive_int>{2_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            reshape_get_output_parallel_dim_degrees(attrs, input);
        ParallelTensorDimDegrees correct = input;

        CHECK(result == correct);
      }

      SUBCASE("does not allow shard degree in core") {
        ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
            SumDegree{1_p},
            DiscardCopyDegree{1_p},
            FFOrdered<positive_int>{1_p, 6_p, 1_p},
        };

        CHECK_THROWS(reshape_get_output_parallel_dim_degrees(attrs, input));
      }
    }

    SUBCASE("changes number of dims") {
      ReshapeAttrs attrs = ReshapeAttrs{
          TensorDims{FFOrdered<positive_int>{6_p, 5_p}},
          TensorDims{FFOrdered<positive_int>{30_p}},
      };

      SUBCASE("input sum degree > 1") {
        ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
            SumDegree{3_p},
            DiscardCopyDegree{1_p},
            FFOrdered<positive_int>{1_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            reshape_get_output_parallel_dim_degrees(attrs, input);
        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            SumDegree{3_p},
            DiscardCopyDegree{1_p},
            FFOrdered<positive_int>{1_p, 1_p},
        };

        CHECK(result == correct);
      }

      SUBCASE("input discard copy degree > 1") {
        ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
            SumDegree{1_p},
            DiscardCopyDegree{2_p},
            FFOrdered<positive_int>{1_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            reshape_get_output_parallel_dim_degrees(attrs, input);
        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            SumDegree{1_p},
            DiscardCopyDegree{2_p},
            FFOrdered<positive_int>{1_p, 1_p},
        };

        CHECK(result == correct);
      }

      SUBCASE("allows leading shard degree") {
        ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
            SumDegree{1_p},
            DiscardCopyDegree{1_p},
            FFOrdered<positive_int>{2_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            reshape_get_output_parallel_dim_degrees(attrs, input);
        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            SumDegree{1_p},
            DiscardCopyDegree{1_p},
            FFOrdered<positive_int>{2_p, 1_p},
        };

        CHECK(result == correct);
      }

      SUBCASE("does not allow shard degree in core") {
        ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
            SumDegree{1_p},
            DiscardCopyDegree{1_p},
            FFOrdered<positive_int>{1_p, 6_p, 1_p},
        };

        CHECK_THROWS(reshape_get_output_parallel_dim_degrees(attrs, input));
      }
    }
  }
