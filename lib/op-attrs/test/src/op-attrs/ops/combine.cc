#include "op-attrs/ops/combine.h"
#include "op-attrs/operator_task_space.h"
#include "test/utils/doctest/fmt/expected.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Combine shape inference") {

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
      ff_dim_t dim = ff_dim_t{2_n};
      int_ge_two degree = 3_ge2;
      CombineAttrs attrs = CombineAttrs{
          /*repartition_dim=*/dim,
          /*repartition_degree=*/degree,
      };

      ParallelTensorShape result =
          combine_get_output_parallel_shape(attrs, input);

      ParallelTensorShape correct = [&] {
        ParallelTensorShape output = input;
        output.dims.shard_dims.at(dim).degree = 1_p;
        return output;
      }();

      CHECK(result == correct);
    }

    SUBCASE("invalid") {
      ff_dim_t dim = ff_dim_t{2_n};
      int_ge_two degree = 4_ge2;
      CombineAttrs attrs = CombineAttrs{
          /*repartition_dim=*/dim,
          /*repartition_degree=*/degree,
      };

      CHECK_THROWS(combine_get_output_parallel_shape(attrs, input));
    }
  }

  TEST_CASE("get_output_parallel_dim_degrees(CombineAttrs, "
            "ParallelTensorDimDegrees)") {
    CombineAttrs attrs = CombineAttrs{
        /*combine_dim=*/ff_dim_t{0_n},
        /*combine_degree=*/3_ge2,
    };

    ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
        SumDegree{2_p},
        DiscardCopyDegree{1_p},
        FFOrdered<positive_int>{
            6_p,
            3_p,
        },
    };

    ParallelTensorDimDegrees result =
        get_output_parallel_dim_degrees(attrs, input_degrees);

    ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
        SumDegree{2_p},
        DiscardCopyDegree{1_p},
        FFOrdered<positive_int>{
            2_p,
            3_p,
        },
    };

    CHECK(result == correct);
  }

  TEST_CASE("get_operator_task_space(CombineAttrs, ParallelTensorDimDegrees)") {
    CombineAttrs attrs = CombineAttrs{
        /*combine_dim=*/ff_dim_t{0_n},
        /*combine_degree=*/3_ge2,
    };

    ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
        SumDegree{2_p},
        DiscardCopyDegree{1_p},
        FFOrdered<positive_int>{
            4_p,
            2_p,
        },
    };

    OperatorTaskSpace result = get_operator_task_space(attrs, input_degrees);
    OperatorTaskSpace correct = operator_task_space_from_minimal_dim_domain(
        MinimalDimDomain<operator_task_space_dim_idx_t>{
            std::map<operator_task_space_dim_idx_t, int_ge_two>{
                {operator_task_space_dim_idx_t{0_n}, 2_ge2},
                {operator_task_space_dim_idx_t{1_n}, 4_ge2},
                {operator_task_space_dim_idx_t{2_n}, 2_ge2},
            },
        });

    CHECK(result == correct);
  }
}
