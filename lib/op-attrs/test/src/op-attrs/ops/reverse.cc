#include <doctest/doctest.h>
#include "op-attrs/ops/reverse.h"
#include "utils/not_implemented.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("reverse_get_output_shape") {
    ReverseAttrs attrs = ReverseAttrs{
      /*axis=*/ff_dim_t{1_n},
    };

    SUBCASE("tensor has axis dim") {
      TensorShape input = TensorShape{
          TensorDims{FFOrdered<positive_int>{4_p, 5_p, 5_p}},
          DataType::FLOAT,
      };

      TensorShape result = reverse_get_output_shape(attrs, input);
      TensorShape correct = input;

      CHECK(result == input);
    }

    SUBCASE("tensor does not have axis dim") {
      TensorShape input = TensorShape{
          TensorDims{FFOrdered<positive_int>{5_p}},
          DataType::FLOAT,
      };

      CHECK_THROWS(reverse_get_output_shape(attrs, input));
    }
  }

  TEST_CASE("reverse_get_output_parallel_dim_degrees") {
    ReverseAttrs attrs = ReverseAttrs{
      /*axis=*/ff_dim_t{1_n},
    };

    SUBCASE("tensor does not have axis dim") {
      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
        /*sum_degree=*/SumDegree{1_p},
        /*discard_copy_degree=*/DiscardCopyDegree{1_p},
        /*shard_degrees=*/FFOrdered<positive_int>{
          1_p,
        }
      };

      CHECK_THROWS(reverse_get_output_parallel_dim_degrees(attrs, input));
    }

    SUBCASE("tensor is parallel in axis dim") {
      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
        /*sum_degree=*/SumDegree{1_p},
        /*discard_copy_degree=*/DiscardCopyDegree{1_p},
        /*shard_degrees=*/FFOrdered<positive_int>{
          1_p, 2_p, 1_p,
        }
      };

      CHECK_THROWS(reverse_get_output_parallel_dim_degrees(attrs, input));
    }

    SUBCASE("tensor is parallel in non-axis dims") {
      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
        /*sum_degree=*/SumDegree{1_p},
        /*discard_copy_degree=*/DiscardCopyDegree{1_p},
        /*shard_degrees=*/FFOrdered<positive_int>{
          3_p, 1_p, 2_p, 1_p,
        }
      };

      ParallelTensorDimDegrees result = reverse_get_output_parallel_dim_degrees(attrs, input);
      ParallelTensorDimDegrees correct = input;

      CHECK(result == input);
    }
  }
}
