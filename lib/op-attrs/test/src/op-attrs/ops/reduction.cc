#include "op-attrs/ops/reduction.h"
#include "test/utils/doctest/fmt/expected.h"
#include <doctest/doctest.h>
#include "op-attrs/operator_task_space.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Reduction shape inference") {

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

      tl::expected<ParallelTensorShape, std::string> result =
          get_output_shape(attrs, input);

      tl::expected<ParallelTensorShape, std::string> correct = [&] {
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

      tl::expected<ParallelTensorShape, std::string> result =
          get_output_shape(attrs, input);

      CHECK_MESSAGE(!result.has_value(),
                    "Unexpected successful result: ",
                    result.error());
    }
  }

  TEST_CASE("get_output_parallel_dim_degrees(ReductionAttrs, ParallelTensorDimDegrees)") {
    ReductionAttrs attrs = ReductionAttrs{
        /*reduction_degree=*/3_ge2,
    };

    ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
      SumDegree{2_p},
      DiscardCopyDegree{6_p},
      FFOrdered<positive_int>{
        1_p,
        3_p,
      },
    };

    ParallelTensorDimDegrees result = get_output_parallel_dim_degrees(attrs, input_degrees);

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

  TEST_CASE("get_operator_task_space(ReductionAttrs, ParallelTensorDimDegrees)") {
    ReductionAttrs attrs = ReductionAttrs{
        /*reduction_degree=*/3_ge2,
    };

    ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
      SumDegree{2_p},
      DiscardCopyDegree{6_p},
      FFOrdered<positive_int>{
        1_p,
        3_p,
      },
    };

    OperatorTaskSpace result = get_operator_task_space(attrs, input_degrees);
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
