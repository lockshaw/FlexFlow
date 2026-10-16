#include "op-attrs/ops/cast.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "test/utils/doctest/fmt/expected.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("cast_get_output_shape") {
    CastAttrs attrs = CastAttrs{DataType::DOUBLE};

    TensorShape input = TensorShape{
        TensorDims{FFOrdered{12_p, 16_p}},
        DataType::FLOAT,
    };

    TensorShape result = cast_get_output_shape(attrs, input);

    TensorShape correct = TensorShape{
        TensorDims{FFOrdered{12_p, 16_p}},
        DataType::DOUBLE,
    };

    CHECK(result == correct);
  }

  TEST_CASE("cast_get_output_parallel_dim_degrees") {
    CastAttrs attrs = CastAttrs{DataType::DOUBLE};

    ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
        SumDegree{2_p},
        DiscardCopyDegree{3_p},
        FFOrdered<positive_int>{
            4_p,
            8_p,
        },
    };

    ParallelTensorDimDegrees result =
        cast_get_output_parallel_dim_degrees(attrs, input);

    ParallelTensorDimDegrees correct = input;

    CHECK(result == correct);
  }

  TEST_CASE("cast_get_output_parallel_shape") {
    // since any edge cases are already tested in
    // cast_get_output_shape and cast_get_output_parallel_dim_degrees,
    // here we just do a basic check that they compose

    CastAttrs attrs = CastAttrs{DataType::DOUBLE};

    ParallelTensorShape input = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{48_p, 4_p},
                ShardParallelDim{56_p, 2_p},
            },
            ReplicaParallelDimSet{
                SumDegree{5_p},
                DiscardCopyDegree{3_p},
            },
        },
        DataType::FLOAT,
    };

    ParallelTensorShape result = cast_get_output_parallel_shape(attrs, input);

    ParallelTensorShape correct = [&] {
      ParallelTensorShape result = input;
      result.data_type = DataType::DOUBLE;
      return result;
    }();

    CHECK(result == correct);
  }
}
