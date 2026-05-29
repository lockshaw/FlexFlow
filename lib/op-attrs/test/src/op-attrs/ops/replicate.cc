#include "op-attrs/ops/replicate.h"
#include "op-attrs/operator_task_space.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Replicate shape inference") {
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

    ParallelTensorShape result = get_output_shape(attrs, input);

    ParallelTensorShape correct_output = input;
    correct_output.dims.replica_dims.discard_copy_degree =
        DiscardCopyDegree{8_p};

    CHECK(result == correct_output);
  }

  TEST_CASE("replicate_get_output_parallel_dim_degrees(ReplicateAttrs, "
            "ParallelTensorDimDegrees)") {
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

  TEST_CASE("replicate_get_operator_task_space(ReplicateAttrs, "
            "ParallelTensorDimDegrees)") {
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

    OperatorTaskSpace result =
        replicate_get_operator_task_space(attrs, input_degrees);
    OperatorTaskSpace correct = operator_task_space_from_minimal_dim_domain(
        MinimalDimDomain<operator_task_space_dim_idx_t>{
            std::unordered_map<operator_task_space_dim_idx_t, int_ge_two>{
                {operator_task_space_dim_idx_t{0_n}, 2_ge2},
                {operator_task_space_dim_idx_t{1_n}, 6_ge2},
                {operator_task_space_dim_idx_t{2_n}, 3_ge2},
            },
        });

    CHECK(result == correct);
  }
}
