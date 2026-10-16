#include "op-attrs/ops/concat.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "test/utils/doctest/fmt/expected.h"
#include "test/utils/doctest/fmt/optional.h"
#include "utils/expected.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("concat_get_output_shape") {
    ConcatAttrs attrs = ConcatAttrs{
        /*axis=*/ff_dim_t{1_n},
        /*num_inputs=*/3_ge2,
    };

    SUBCASE("empty input shapes list passed") {
      std::vector<TensorShape> input_shapes = {};

      CHECK_THROWS(concat_get_output_shape(attrs, input_shapes));
    }

    positive_int dim0_size = 12_p;
    positive_int dim2_size = 20_p;
    TensorShape input_shape1 = TensorShape{
        TensorDims{FFOrdered{
            dim0_size,
            14_p,
            dim2_size,
        }},
        DataType::FLOAT,
    };

    SUBCASE("single element input shapes list passed") {
      std::vector<TensorShape> input_shapes = {input_shape1};

      CHECK_THROWS(concat_get_output_shape(attrs, input_shapes));
    }

    TensorShape input_shape2 = TensorShape{
        TensorDims{FFOrdered{
            dim0_size,
            16_p,
            dim2_size,
        }},
        DataType::FLOAT,
    };

    TensorShape input_shape3 = TensorShape{
        TensorDims{FFOrdered{dim0_size, 18_p, dim2_size}},
        DataType::FLOAT,
    };

    SUBCASE("input shapes do not share the same num_dims") {
      TensorShape mismatched_num_dims = TensorShape{
          TensorDims{FFOrdered{
              dim0_size,
              20_p,
              dim2_size,
              1_p,
          }},
          DataType::FLOAT,
      };

      std::vector<TensorShape> input_shapes = {
          input_shape1, input_shape2, input_shape3, mismatched_num_dims};

      CHECK_THROWS(concat_get_output_shape(attrs, input_shapes));
    }

    SUBCASE("concat axis is out of bounds") {
      attrs = ConcatAttrs{
          /*axis=*/ff_dim_t{3_n},
          /*num_inputs=*/3_ge2,
      };

      std::vector<TensorShape> input_shapes = {
          input_shape1, input_shape2, input_shape3};

      CHECK_THROWS(concat_get_output_shape(attrs, input_shapes));
    }

    SUBCASE("input shapes are valid") {
      std::vector<TensorShape> input_shapes = {
          input_shape1, input_shape2, input_shape3};

      TensorShape result = concat_get_output_shape(attrs, input_shapes);
      TensorShape correct = TensorShape{
          TensorDims{FFOrdered{
              dim0_size,
              14_p + 16_p + 18_p,
              dim2_size,
          }},
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("concat_get_output_parallel_dim_degrees") {
    ConcatAttrs attrs = ConcatAttrs{
        /*axis=*/ff_dim_t{1_n},
        /*num_inputs=*/3_ge2,
    };

    auto mk_3d = [&](int o_sum, int o_eq, int o0, int o1, int o2) {
      return ParallelTensorDimDegrees{
          SumDegree{positive_int{o_sum}},
          DiscardCopyDegree{positive_int{o_eq}},
          FFOrdered<positive_int>{
              positive_int{o0},
              positive_int{o1},
              positive_int{o2},
          },
      };
    };

    SUBCASE("sum reduction parallelism") {
      SUBCASE("matching") {
        SumDegree sum_degree = SumDegree{2_p};

        std::vector<ParallelTensorDimDegrees> inputs = {
            mk_3d(2, 1, 1, 1, 1),
            mk_3d(2, 1, 1, 1, 1),
            mk_3d(2, 1, 1, 1, 1),
        };

        ParallelTensorDimDegrees result =
            concat_get_output_parallel_dim_degrees(attrs, inputs);
        ParallelTensorDimDegrees correct = mk_3d(2, 1, 1, 1, 1);

        CHECK(result == correct);
      }

      SUBCASE("not matching") {
        std::vector<ParallelTensorDimDegrees> inputs = {
            mk_3d(2, 1, 1, 1, 1),
            mk_3d(4, 1, 1, 1, 1),
            mk_3d(4, 1, 1, 1, 1),
        };

        CHECK_THROWS(concat_get_output_parallel_dim_degrees(attrs, inputs));
      }
    }

    SUBCASE("discard copy reduction parallelism") {
      SUBCASE("matching") {
        DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{2_p};

        std::vector<ParallelTensorDimDegrees> inputs = {
            mk_3d(1, 2, 1, 1, 1),
            mk_3d(1, 2, 1, 1, 1),
            mk_3d(1, 2, 1, 1, 1),
        };

        ParallelTensorDimDegrees result =
            concat_get_output_parallel_dim_degrees(attrs, inputs);
        ParallelTensorDimDegrees correct = mk_3d(1, 2, 1, 1, 1);

        CHECK(result == correct);
      }

      SUBCASE("not matching") {
        std::vector<ParallelTensorDimDegrees> inputs = {
            mk_3d(1, 2, 1, 1, 1),
            mk_3d(1, 2, 1, 1, 1),
            mk_3d(1, 4, 1, 1, 1),
        };

        CHECK_THROWS(concat_get_output_parallel_dim_degrees(attrs, inputs));
      }
    }

    SUBCASE("parallelism in axis dim") {
      SUBCASE("matching") {
        positive_int degree = 2_p;

        std::vector<ParallelTensorDimDegrees> inputs = {
            mk_3d(1, 1, 1, 2, 1),
            mk_3d(1, 1, 1, 2, 1),
            mk_3d(1, 1, 1, 2, 1),
        };

        CHECK_THROWS(concat_get_output_parallel_dim_degrees(attrs, inputs));
      }

      SUBCASE("not matching") {
        std::vector<ParallelTensorDimDegrees> inputs = {
            mk_3d(1, 1, 1, 1, 1),
            mk_3d(1, 1, 1, 1, 1),
            mk_3d(1, 1, 1, 2, 1),
        };

        CHECK_THROWS(concat_get_output_parallel_dim_degrees(attrs, inputs));
      }
    }

    SUBCASE("parallelism in non-axis shard dims") {
      SUBCASE("matching") {
        positive_int degree0 = 2_p;
        positive_int degree2 = 4_p;

        std::vector<ParallelTensorDimDegrees> inputs = {
            mk_3d(1, 1, 2, 1, 4),
            mk_3d(1, 1, 2, 1, 4),
            mk_3d(1, 1, 2, 1, 4),
        };

        ParallelTensorDimDegrees result =
            concat_get_output_parallel_dim_degrees(attrs, inputs);
        ParallelTensorDimDegrees correct = mk_3d(1, 1, 2, 1, 4);

        CHECK(result == correct);
      }

      SUBCASE("not matching") {
        std::vector<ParallelTensorDimDegrees> inputs = {
            mk_3d(1, 1, 2, 1, 4),
            mk_3d(1, 1, 4, 1, 2),
            mk_3d(1, 1, 4, 1, 2),
        };

        CHECK_THROWS(concat_get_output_parallel_dim_degrees(attrs, inputs));
      }
    }

    SUBCASE("parallelism degrees are not mutually exclusive") {
      std::vector<ParallelTensorDimDegrees> inputs = {
          mk_3d(3, 5, 2, 1, 4),
          mk_3d(3, 5, 2, 1, 4),
          mk_3d(3, 5, 2, 1, 4),
      };

      ParallelTensorDimDegrees result =
          concat_get_output_parallel_dim_degrees(attrs, inputs);
      ParallelTensorDimDegrees correct = mk_3d(3, 5, 2, 1, 4);

      CHECK(result == correct);
    }
  }

  TEST_CASE("concat_get_output_parallel_shape") {
    // since most of the edge cases are already tested in
    // concat_get_output_shape and concat_get_output_parallel_dim_degrees,
    // here we just do a basic check that they compose

    ConcatAttrs attrs = ConcatAttrs{
        /*axis=*/ff_dim_t{1_n},
        /*num_inputs=*/4_ge2,
    };

    auto mk_input_of_size = [](int x) -> ParallelTensorShape {
      return ParallelTensorShape{
          ParallelTensorDims{
              FFOrdered<ShardParallelDim>{
                  ShardParallelDim{12_p, 2_p},
                  ShardParallelDim{positive_int{x}, 1_p},
                  ShardParallelDim{20_p, 4_p},
              },
              ReplicaParallelDimSet{
                  SumDegree{3_p},
                  DiscardCopyDegree{5_p},
              },
          },
          DataType::INT32,
      };
    };

    std::vector<ParallelTensorShape> inputs = {
        mk_input_of_size(14),
        mk_input_of_size(16),
        mk_input_of_size(14),
        mk_input_of_size(21),
    };

    ParallelTensorShape result =
        concat_get_output_parallel_shape(attrs, inputs);

    ParallelTensorShape correct = mk_input_of_size(14 + 16 + 14 + 21);

    CHECK(result == correct);
  }
}
