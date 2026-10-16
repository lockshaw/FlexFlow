#include "op-attrs/ops/input.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("input_get_output_shape") {
    TensorShape shape = TensorShape{
        TensorDims{
            FFOrdered<positive_int>{
                12_p,
                8_p,
                13_p,
            },
        },
        DataType::FLOAT,
    };

    InputAttrs attrs = InputAttrs{
        shape,
    };

    TensorShape result = input_get_output_shape(attrs);
    TensorShape correct = shape;

    CHECK(result == correct);
  }

  TEST_CASE("input_get_output_parallel_dim_degrees") {
    TensorShape shape = TensorShape{
        TensorDims{
            FFOrdered<positive_int>{
                12_p,
                8_p,
            },
        },
        DataType::FLOAT,
    };

    InputAttrs attrs = InputAttrs{
        shape,
    };

    ParallelTensorDimDegrees result =
        input_get_output_parallel_dim_degrees(attrs);
    ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
        SumDegree{1_p},
        DiscardCopyDegree{1_p},
        FFOrdered<positive_int>{
            1_p,
            1_p,
        },
    };

    CHECK(result == correct);
  }

  TEST_CASE("input_get_output_parallel_shape") {
    TensorShape shape = TensorShape{
        TensorDims{
            FFOrdered<positive_int>{
                12_p,
                8_p,
                13_p,
            },
        },
        DataType::FLOAT,
    };

    InputAttrs attrs = InputAttrs{
        shape,
    };

    ParallelTensorShape result = input_get_output_parallel_shape(attrs);
    ParallelTensorShape correct = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{12_p, 1_p},
                ShardParallelDim{8_p, 1_p},
                ShardParallelDim{13_p, 1_p},
            },
            ReplicaParallelDimSet{
                SumDegree{1_p},
                DiscardCopyDegree{1_p},
            },
        },
        DataType::FLOAT,
    };

    CHECK(result == correct);
  }
}
