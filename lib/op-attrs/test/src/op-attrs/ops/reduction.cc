#include "op-attrs/ops/reduction.h"
#include "test/utils/doctest/fmt/expected.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("reduction_get_output_parallel_shape") {

    ParallelTensorShape input = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{12_p, 2_p},
                ShardParallelDim{14_p, 1_p},
                ShardParallelDim{16_p, 3_p},
                ShardParallelDim{18_p, 2_p},
            },
            ReplicaParallelDimSet{
                SumDegree{3_p},
                DiscardCopyDegree{2_p},
            },
        },
        DataType::FLOAT,
    };

    SUBCASE("valid") {
      ReductionAttrs attrs = ReductionAttrs{
          /*repartition_degree=*/3_ge2,
      };

      ParallelTensorShape result =
          reduction_get_output_parallel_shape(attrs, input);

      ParallelTensorShape correct = [&] {
        ParallelTensorShape output = input;
        output.dims.replica_dims.sum_degree.value = 1_p;
        return output;
      }();

      CHECK(result == correct);
    }

    SUBCASE("invalid") {
      ReductionAttrs attrs = ReductionAttrs{
          /*repartition_degree=*/4_ge2,
      };

      CHECK_THROWS(reduction_get_output_parallel_shape(attrs, input));
    }
  }

  TEST_CASE("reduction_get_output_parallel_dim_degrees") {
    ReductionAttrs attrs = ReductionAttrs{
        /*reduction_degree=*/3_ge2,
    };

    ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
        SumDegree{6_p},
        DiscardCopyDegree{2_p},
        FFOrdered<positive_int>{
            1_p,
            3_p,
        },
    };

    ParallelTensorDimDegrees result =
        reduction_get_output_parallel_dim_degrees(attrs, input_degrees);

    ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
        SumDegree{2_p},
        DiscardCopyDegree{2_p},
        FFOrdered<positive_int>{
            1_p,
            3_p,
        },
    };

    CHECK(result == correct);
  }
}
