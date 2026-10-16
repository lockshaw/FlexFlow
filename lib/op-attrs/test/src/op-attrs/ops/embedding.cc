#include "op-attrs/ops/embedding.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "test/utils/doctest/fmt/expected.h"
#include "utils/integer_conversions.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("embedding_get_output_shape") {
    EmbeddingAttrs attrs = EmbeddingAttrs{
        /*num_entries=*/128_p,
        /*out_channels=*/1024_p,
        /*aggr=*/AggregateOp::SUM,
        /*data_type=*/DataType::FLOAT,
    };

    TensorShape input = TensorShape{
        TensorDims{
            FFOrdered{
                48_p,
                56_p,
            },
        },
        DataType::INT32,
    };

    TensorShape result = embedding_get_output_shape(attrs, input);

    TensorShape correct = TensorShape{
        TensorDims{
            FFOrdered{
                48_p,
                1024_p,
            },
        },
        DataType::FLOAT,
    };

    CHECK(result == correct);
  }

  TEST_CASE("embedding_get_weights_shape") {
    EmbeddingAttrs attrs = EmbeddingAttrs{
        /*num_entries=*/128_p,
        /*out_channels=*/1024_p,
        /*aggr=*/AggregateOp::SUM,
        /*data_type=*/DataType::FLOAT,
    };

    TensorShape input = TensorShape{
        TensorDims{
            FFOrdered{
                48_p,
                56_p,
            },
        },
        DataType::INT32,
    };

    TensorShape result = embedding_get_weights_shape(attrs, input);

    TensorShape correct = TensorShape{
        TensorDims{
            FFOrdered{
                128_p,
                1024_p,
            },
        },
        DataType::FLOAT,
    };

    CHECK(result == correct);
  }

  TEST_CASE("embedding_get_output_parallel_dim_degrees") {
    EmbeddingAttrs attrs = EmbeddingAttrs{
        /*num_entries=*/128_p,
        /*out_channels=*/1024_p,
        /*aggr=*/AggregateOp::SUM,
        /*data_type=*/DataType::FLOAT,
    };

    SUBCASE("data parallelism") {
      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
          SumDegree{1_p},
          DiscardCopyDegree{1_p},
          FFOrdered{4_p, 1_p},
      };

      ParallelTensorDimDegrees result =
          embedding_get_output_parallel_dim_degrees(attrs, input);

      ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
          SumDegree{1_p},
          DiscardCopyDegree{1_p},
          FFOrdered{4_p, 1_p},
      };

      CHECK(result == correct);
    }

    SUBCASE("input features parallelism") {
      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
          SumDegree{1_p},
          DiscardCopyDegree{1_p},
          FFOrdered{1_p, 4_p},
      };

      ParallelTensorDimDegrees result =
          embedding_get_output_parallel_dim_degrees(attrs, input);

      ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
          SumDegree{4_p},
          DiscardCopyDegree{1_p},
          FFOrdered{1_p, 1_p},
      };

      CHECK(result == correct);
    }

    SUBCASE("output channel shard parallelism") {
      // NOTE (@lockshaw): in the current (parallel shape inference from just
      // input tensor) representation we have to choose between either
      // parallelism in the weight channel dimension or in the weight entry
      // dimension. For now we choose to represent parallelism in the channel
      // dimension, but partitioning in the entry dimension is also potentially
      // useful as it produces sum parallelism in the output
      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
          SumDegree{1_p},
          DiscardCopyDegree{4_p},
          FFOrdered{1_p, 1_p},
      };

      ParallelTensorDimDegrees result =
          embedding_get_output_parallel_dim_degrees(attrs, input);

      ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
          SumDegree{1_p},
          DiscardCopyDegree{1_p},
          FFOrdered{1_p, 4_p},
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("embedding_get_weights_parallel_dim_degrees") {
    EmbeddingAttrs attrs = EmbeddingAttrs{
        /*num_entries=*/128_p,
        /*out_channels=*/1024_p,
        /*aggr=*/AggregateOp::SUM,
        /*data_type=*/DataType::FLOAT,
    };

    SUBCASE("data parallelism") {
      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
          SumDegree{1_p},
          DiscardCopyDegree{1_p},
          FFOrdered{4_p, 1_p},
      };

      ParallelTensorDimDegrees result =
          embedding_get_weights_parallel_dim_degrees(attrs, input);

      ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
          SumDegree{1_p},
          DiscardCopyDegree{4_p},
          FFOrdered{1_p, 1_p},
      };

      CHECK(result == correct);
    }

    SUBCASE("input features parallelism") {
      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
          SumDegree{1_p},
          DiscardCopyDegree{1_p},
          FFOrdered{1_p, 4_p},
      };

      ParallelTensorDimDegrees result =
          embedding_get_weights_parallel_dim_degrees(attrs, input);

      ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
          SumDegree{1_p},
          DiscardCopyDegree{4_p},
          FFOrdered{1_p, 1_p},
      };

      CHECK(result == correct);
    }

    SUBCASE("output channel shard parallelism") {
      ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
          SumDegree{1_p},
          DiscardCopyDegree{4_p},
          FFOrdered{1_p, 1_p},
      };

      ParallelTensorDimDegrees result =
          embedding_get_weights_parallel_dim_degrees(attrs, input);

      ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
          SumDegree{1_p},
          DiscardCopyDegree{1_p},
          FFOrdered{1_p, 4_p},
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("embedding_get_output_parallel_shape") {
    // since most of the edge cases are already tested in
    // embedding_get_output_shape and embedding_get_output_parallel_dim_degrees,
    // here we just do a basic check that they compose

    EmbeddingAttrs attrs = EmbeddingAttrs{
        /*num_entries=*/128_p,
        /*out_channels=*/600_p,
        /*aggr=*/AggregateOp::SUM,
        /*data_type=*/DataType::FLOAT,
    };

    ParallelTensorShape input_shape = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{48_p, 4_p},
                ShardParallelDim{56_p, 2_p},
            },
            ReplicaParallelDimSet{
                SumDegree{1_p},
                DiscardCopyDegree{3_p},
            },
        },
        DataType::INT32,
    };

    ParallelTensorShape result =
        embedding_get_output_parallel_shape(attrs, input_shape);

    ParallelTensorShape correct = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{48_p, 4_p},
                ShardParallelDim{600_p, 3_p},
            },
            ReplicaParallelDimSet{
                SumDegree{2_p},
                DiscardCopyDegree{1_p},
            },
        },
        DataType::FLOAT,
    };

    CHECK(result == correct);
  }

  TEST_CASE("embedding_get_weights_parallel_shape") {
    // since most of the edge cases are already tested in
    // embedding_get_weights_shape and embedding_get_weights_parallel_dim_degrees,
    // here we just do a basic check that they compose

    EmbeddingAttrs attrs = EmbeddingAttrs{
        /*num_entries=*/128_p,
        /*out_channels=*/600_p,
        /*aggr=*/AggregateOp::SUM,
        /*data_type=*/DataType::FLOAT,
    };

    ParallelTensorShape input_shape = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{48_p, 4_p},
                ShardParallelDim{56_p, 2_p},
            },
            ReplicaParallelDimSet{
                SumDegree{1_p},
                DiscardCopyDegree{3_p},
            },
        },
        DataType::INT32,
    };

    ParallelTensorShape result =
        embedding_get_weights_parallel_shape(attrs, input_shape);

    ParallelTensorShape correct = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{128_p, 1_p},
                ShardParallelDim{600_p, 3_p},
            },
            ReplicaParallelDimSet{
                SumDegree{1_p},
                DiscardCopyDegree{8_p},
            },
        },
        DataType::FLOAT,
    };

    CHECK(result == correct);
  }
}
