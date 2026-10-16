#include "op-attrs/ops/noop.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("noop_get_output_shape") {
    TensorShape input = TensorShape{
        TensorDims{
            FFOrdered<positive_int>{
                12_p,
                8_p,
                13_p,
            },
        },
        DataType::FLOAT,
    };

    NoopAttrs attrs = NoopAttrs{};

    TensorShape result = noop_get_output_shape(attrs, input);
    TensorShape correct = input;

    CHECK(result == correct);
  }

  TEST_CASE("noop_get_output_parallel_dim_degrees") {
    ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
        SumDegree{3_p},
        DiscardCopyDegree{5_p},
        FFOrdered<positive_int>{
            6_p,
            4_p,
        },
    };

    NoopAttrs attrs = NoopAttrs{};

    ParallelTensorDimDegrees result =
        noop_get_output_parallel_dim_degrees(attrs, input);
    ParallelTensorDimDegrees correct = input;

    CHECK(result == correct);
  }

  TEST_CASE("noop_get_output_parallel_shape") {
    ParallelTensorShape input = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{12_p, 3_p},
                ShardParallelDim{8_p, 2_p},
                ShardParallelDim{13_p, 1_p},
            },
            ReplicaParallelDimSet{
                SumDegree{5_p},
                DiscardCopyDegree{7_p},
            },
        },
        DataType::FLOAT,
    };

    NoopAttrs attrs = NoopAttrs{};

    ParallelTensorShape result = noop_get_output_parallel_shape(attrs, input);
    ParallelTensorShape correct = input;

    CHECK(result == correct);
  }
}
