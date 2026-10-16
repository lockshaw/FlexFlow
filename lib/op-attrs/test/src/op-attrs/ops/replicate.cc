#include "op-attrs/ops/replicate.h"
#include "op-attrs/operator_task_space.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("replicate_get_output_parallel_shape") {
    ReplicateAttrs attrs = ReplicateAttrs{
        /*replicate_degree=*/4_ge2,
    };

    ParallelTensorShape input = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{10_p, 2_p},
                ShardParallelDim{12_p, 1_p},
                ShardParallelDim{14_p, 2_p},
                ShardParallelDim{16_p, 2_p},
            },
            ReplicaParallelDimSet{
                SumDegree{3_p},
                DiscardCopyDegree{2_p},
            },
        },
        DataType::FLOAT,
    };

    ParallelTensorShape result =
        replicate_get_output_parallel_shape(attrs, input);

    ParallelTensorShape correct_output = input;
    correct_output.dims.replica_dims.discard_copy_degree =
        DiscardCopyDegree{8_p};

    CHECK(result == correct_output);
  }

  TEST_CASE("replicate_get_output_parallel_dim_degrees") {
    ReplicateAttrs attrs = ReplicateAttrs{
        /*replicate_degree=*/3_ge2,
    };

    ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
        SumDegree{2_p},
        DiscardCopyDegree{2_p},
        FFOrdered<positive_int>{
            1_p,
            3_p,
        },
    };

    ParallelTensorDimDegrees result =
        replicate_get_output_parallel_dim_degrees(attrs, input_degrees);

    ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
        SumDegree{2_p},
        DiscardCopyDegree{6_p},
        FFOrdered<positive_int>{
            1_p,
            3_p,
        },
    };

    CHECK(result == correct);
  }
}
