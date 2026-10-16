#include "op-attrs/ops/reduce.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("reduce_get_output_shape") {
    TensorShape input = TensorShape{
        TensorDims{
            FFOrdered<positive_int>{
                8_p,
                12_p,
                6_p,
                2_p,
            },
        },
        DataType::FLOAT,
    };

    SUBCASE("keepdims = true") {
      ReduceAttrs attrs = ReduceAttrs{
          /*axes=*/nonempty_set<ff_dim_t>{
              ff_dim_t{1_n},
              ff_dim_t{3_n},
          },
          /*op=*/ReduceOp::SUM,
          /*keepdims=*/true,
      };

      TensorShape result = reduce_get_output_shape(attrs, input);
      TensorShape correct = input;

      CHECK(result == correct);
    }

    SUBCASE("keepdims = false") {
      ReduceAttrs attrs = ReduceAttrs{
          /*axes=*/nonempty_set<ff_dim_t>{
              ff_dim_t{1_n},
              ff_dim_t{3_n},
          },
          /*op=*/ReduceOp::SUM,
          /*keepdims=*/false,
      };

      TensorShape result = reduce_get_output_shape(attrs, input);
      TensorShape correct = TensorShape{
          TensorDims{
              FFOrdered<positive_int>{
                  8_p,
                  6_p,
              },
          },
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("reduce_get_output_parallel_dim_degrees") {
    auto mk_2d = [&](int o_sum,
                     int o_discard,
                     int o_0,
                     int o_1) -> ParallelTensorDimDegrees {
      return ParallelTensorDimDegrees{
          SumDegree{positive_int{o_sum}},
          DiscardCopyDegree{positive_int{o_discard}},
          FFOrdered<positive_int>{
              positive_int{o_0},
              positive_int{o_1},
          },
      };
    };

    auto mk_4d =
        [&](int o_sum, int o_discard, int o_0, int o_1, int o_2, int o_3)
        -> ParallelTensorDimDegrees {
      return ParallelTensorDimDegrees{
          SumDegree{positive_int{o_sum}},
          DiscardCopyDegree{positive_int{o_discard}},
          FFOrdered<positive_int>{
              positive_int{o_0},
              positive_int{o_1},
              positive_int{o_2},
              positive_int{o_3},
          },
      };
    };

    auto mk_attrs = [&](ReduceOp op, bool keepdims = true) -> ReduceAttrs {
      return ReduceAttrs{
          /*axes=*/nonempty_set<ff_dim_t>{
              ff_dim_t{1_n},
              ff_dim_t{3_n},
          },
          /*op=*/op,
          /*keepdims=*/keepdims,
      };
    };

    ReduceAttrs sum_attrs = mk_attrs(ReduceOp::SUM);
    ReduceAttrs max_attrs = mk_attrs(ReduceOp::AVG);

    SUBCASE("pre-existing sum parallelism") {
      ParallelTensorDimDegrees input = mk_4d(3, 1, 1, 1, 1, 1);

      SUBCASE("op = SUM") {
        ParallelTensorDimDegrees result =
            reduce_get_output_parallel_dim_degrees(sum_attrs, input);
        ParallelTensorDimDegrees correct = mk_4d(3, 1, 1, 1, 1, 1);

        CHECK(result == correct);
      }

      SUBCASE("op = AVG") {
        CHECK_THROWS(reduce_get_output_parallel_dim_degrees(max_attrs, input));
      }
    }

    SUBCASE("pre-existing discard copy parallelism") {
      ParallelTensorDimDegrees input = mk_4d(1, 3, 1, 1, 1, 1);

      SUBCASE("op = SUM") {
        ParallelTensorDimDegrees result =
            reduce_get_output_parallel_dim_degrees(sum_attrs, input);
        ParallelTensorDimDegrees correct = mk_4d(1, 3, 1, 1, 1, 1);

        CHECK(result == correct);
      }

      SUBCASE("op = AVG") {
        ParallelTensorDimDegrees result =
            reduce_get_output_parallel_dim_degrees(max_attrs, input);
        ParallelTensorDimDegrees correct = mk_4d(1, 3, 1, 1, 1, 1);

        CHECK(result == correct);
      }
    }

    SUBCASE("shard parallelism not in reduction axes") {
      ParallelTensorDimDegrees input = mk_4d(1, 1, 4, 1, 1, 1);

      SUBCASE("op = SUM") {
        ParallelTensorDimDegrees result =
            reduce_get_output_parallel_dim_degrees(sum_attrs, input);
        ParallelTensorDimDegrees correct = mk_4d(1, 1, 4, 1, 1, 1);

        CHECK(result == correct);
      }

      SUBCASE("op = AVG") {
        ParallelTensorDimDegrees result =
            reduce_get_output_parallel_dim_degrees(max_attrs, input);
        ParallelTensorDimDegrees correct = mk_4d(1, 1, 4, 1, 1, 1);

        CHECK(result == correct);
      }
    }

    SUBCASE("shard parallelism in reduction axis") {
      ParallelTensorDimDegrees input = mk_4d(1, 1, 1, 4, 1, 1);

      SUBCASE("op = SUM") {
        CHECK_THROWS(reduce_get_output_parallel_dim_degrees(sum_attrs, input));
      }

      SUBCASE("op = AVG") {
        CHECK_THROWS(reduce_get_output_parallel_dim_degrees(max_attrs, input));
      }
    }

    SUBCASE("hybrid parallelism") {
      SUBCASE("op = SUM") {
        ParallelTensorDimDegrees input = mk_4d(3, 7, 2, 1, 5, 1);

        ParallelTensorDimDegrees result =
            reduce_get_output_parallel_dim_degrees(sum_attrs, input);
        ParallelTensorDimDegrees correct = mk_4d(3, 7, 2, 1, 5, 1);

        CHECK(result == correct);
      }

      SUBCASE("op = AVG") {
        ParallelTensorDimDegrees input = mk_4d(1, 7, 2, 1, 5, 1);

        ParallelTensorDimDegrees result =
            reduce_get_output_parallel_dim_degrees(max_attrs, input);
        ParallelTensorDimDegrees correct = mk_4d(1, 7, 2, 1, 5, 1);

        CHECK(result == correct);
      }
    }

    SUBCASE("keepdims = false") {
      SUBCASE("op = SUM") {
        ReduceAttrs attrs = mk_attrs(ReduceOp::SUM, /*keepdims=*/false);

        ParallelTensorDimDegrees input = mk_4d(3, 7, 2, 1, 5, 1);

        ParallelTensorDimDegrees result =
            reduce_get_output_parallel_dim_degrees(attrs, input);
        ParallelTensorDimDegrees correct = mk_2d(3, 7, 2, 5);

        CHECK(result == correct);
      }

      SUBCASE("op = AVG") {
        ReduceAttrs attrs = mk_attrs(ReduceOp::AVG, /*keepdims=*/false);

        ParallelTensorDimDegrees input = mk_4d(1, 7, 2, 1, 5, 1);

        ParallelTensorDimDegrees result =
            reduce_get_output_parallel_dim_degrees(attrs, input);
        ParallelTensorDimDegrees correct = mk_2d(1, 7, 2, 5);

        CHECK(result == correct);
      }
    }
  }

  TEST_CASE("reduce_get_output_parallel_shape") {
    // since most of the edge cases are already tested in
    // reduce_get_output_shape and reduce_get_output_parallel_dim_degrees,
    // here we just do a basic check that they compose

    ReduceAttrs attrs = ReduceAttrs{
        /*axes=*/nonempty_set<ff_dim_t>{
            ff_dim_t{1_n},
            ff_dim_t{3_n},
        },
        /*op=*/ReduceOp::SUM,
        /*keepdims=*/false,
    };

    ParallelTensorShape input_shape = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{6_p, 2_p},
                ShardParallelDim{7_p, 1_p},
                ShardParallelDim{10_p, 5_p},
                ShardParallelDim{4_p, 1_p},
            },
            ReplicaParallelDimSet{
                SumDegree{7_p},
                DiscardCopyDegree{3_p},
            },
        },
        DataType::FLOAT,
    };

    ParallelTensorShape result =
        reduce_get_output_parallel_shape(attrs, input_shape);

    ParallelTensorShape correct = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{6_p, 2_p},
                ShardParallelDim{10_p, 5_p},
            },
            ReplicaParallelDimSet{
                SumDegree{7_p},
                DiscardCopyDegree{3_p},
            },
        },
        DataType::FLOAT,
    };

    CHECK(result == correct);
  }
}
