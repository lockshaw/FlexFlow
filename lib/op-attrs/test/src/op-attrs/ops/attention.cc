#include "op-attrs/ops/attention.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "test/utils/doctest/fmt/expected.h"
#include "utils/integer_conversions.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

static mk_1d_dim_degrees(int sum_degree, 
                         int discard_copy_degree,
                         int inner_degree) 
      -> ParallelTensorDimDegrees
{
  return ParallelTensorDimDegrees{
    SumDegree{positive_int{sum_degree}},
    DiscardCopyDegree{positive_int{discard_copy_degree}},
    FFOrdered<positive_int>{
      positive_int{inner_degree},
    },
  };
}

static mk_2d_dim_degrees(int sum_degree, 
                         int discard_copy_degree,
                         int embedding_dim_degree,
                         int head_dim_degree) 
      -> ParallelTensorDimDegrees
{
  return ParallelTensorDimDegrees{
    SumDegree{positive_int{sum_degree}},
    DiscardCopyDegree{positive_int{discard_copy_degree}},
    FFOrdered<positive_int>{
      positive_int{embedding_dim_degree},
      positive_int{head_dim_degree},
    },
  };
}

static mk_3d_dim_degrees(int sum_degree, 
                         int discard_copy_degree,
                         int batch_degree,
                         int sequence_degree,
                         int inner_degree) 
      -> ParallelTensorDimDegrees
{
  return ParallelTensorDimDegrees{
    SumDegree{positive_int{sum_degree}},
    DiscardCopyDegree{positive_int{discard_copy_degree}},
    FFOrdered<positive_int>{
      positive_int{batch_degree},
      positive_int{sequence_degree},
      positive_int{inner_degree},
    },
  };
}

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_attention_incoming_tensor_roles") {
    auto make_attrs = [](bool bias) {
      return MultiHeadAttentionAttrs{
          /*embed_dim=*/32_p,
          /*num_heads=*/10_p,
          /*kdim=*/32_p,
          /*vdim=*/32_p,
          /*dropout=*/0.0,
          /*bias=*/bias,
          /*add_bias_kv=*/false,
          /*add_zero_attn=*/false,
      };
    };

    SUBCASE("without bias") {
      MultiHeadAttentionAttrs attrs = make_attrs(/*bias=*/false);

      std::map<TensorSlotName, IncomingTensorRole> result =
          get_attention_incoming_tensor_roles(attrs);
      std::map<TensorSlotName, IncomingTensorRole> correct =
          std::map<TensorSlotName, IncomingTensorRole>{
              {
                  TensorSlotName::KEY,
                  IncomingTensorRole::INPUT,
              },
              {
                  TensorSlotName::QUERY,
                  IncomingTensorRole::INPUT,
              },
              {
                  TensorSlotName::VALUE,
                  IncomingTensorRole::INPUT,
              },
              {
                  TensorSlotName::WEIGHT,
                  IncomingTensorRole::WEIGHT,
              },
          };

      CHECK(result == correct);
    }

    SUBCASE("with bias") {
      MultiHeadAttentionAttrs attrs = make_attrs(/*bias=*/true);

      std::map<TensorSlotName, IncomingTensorRole> result =
          get_attention_incoming_tensor_roles(attrs);
      std::map<TensorSlotName, IncomingTensorRole> correct =
          std::map<TensorSlotName, IncomingTensorRole>{
              {
                  TensorSlotName::KEY,
                  IncomingTensorRole::INPUT,
              },
              {
                  TensorSlotName::QUERY,
                  IncomingTensorRole::INPUT,
              },
              {
                  TensorSlotName::VALUE,
                  IncomingTensorRole::INPUT,
              },
              {
                  TensorSlotName::WEIGHT,
                  IncomingTensorRole::WEIGHT,
              },
              {
                  TensorSlotName::INPUT_BIAS,
                  IncomingTensorRole::WEIGHT,
              },
              {
                  TensorSlotName::OUTPUT_BIAS,
                  IncomingTensorRole::WEIGHT,
              },
          };

      CHECK(result == correct);
    }
  }


  TEST_CASE("attention_get_weights_shape") {
    MultiHeadAttentionAttrs attrs = MultiHeadAttentionAttrs{
        /*embed_dim=*/32_p,
        /*num_heads=*/10_p,
        /*kdim=*/32_p,
        /*vdim=*/32_p,
        /*dropout=*/0.0,
        /*bias=*/true,
        /*add_bias_kv=*/false,
        /*add_zero_attn=*/false,
    };

    TensorShape input_q = TensorShape{
        TensorDims{
            FFOrdered{
                40_p,
                48_p,
                36_p,
            },
        },
        DataType::FLOAT,
    };

    Tensorshape input_k = input_q;
    Tensorshape input_v = input_q;

    TensorShape result =
        attention_get_weights_shape(attrs, input_q, input_k, input_v);

    TensorShape correct = 
      TensorShape{
        TensorDims{
            FFOrdered{
                (feature_size * embed_dim) * 3_p + (embed_dim * embed_dim),
                num_heads,
            },
        },
        DataType::FLOAT,
      };

    CHECK(result == correct);
  }

  TEST_CASE("attention_get_input_bias_shape") {
    MultiHeadAttentionAttrs attrs = MultiHeadAttentionAttrs{
        /*embed_dim=*/32_p,
        /*num_heads=*/10_p,
        /*kdim=*/32_p,
        /*vdim=*/32_p,
        /*dropout=*/0.0,
        /*bias=*/true,
        /*add_bias_kv=*/false,
        /*add_zero_attn=*/false,
    };

    TensorShape input_q = TensorShape{
        TensorDims{
            FFOrdered{
                40_p,
                48_p,
                36_p,
            },
        },
        DataType::FLOAT,
    };

    Tensorshape input_k = input_q;
    Tensorshape input_v = input_q;

    TensorShape result =
        attention_get_input_bias_shape(attrs, input_q, input_k, input_v);

    TensorShape correct = 
      TensorShape{
        TensorDims{
            FFOrdered{
                32_p * 3_p,
            },
        },
        DataType::FLOAT,
    };

    CHECK(result == correct);
  }

  TEST_CASE("attention_get_output_bias_shape") {
    MultiHeadAttentionAttrs attrs = MultiHeadAttentionAttrs{
        /*embed_dim=*/32_p,
        /*num_heads=*/10_p,
        /*kdim=*/32_p,
        /*vdim=*/32_p,
        /*dropout=*/0.0,
        /*bias=*/true,
        /*add_bias_kv=*/false,
        /*add_zero_attn=*/false,
    };

    TensorShape input_q = TensorShape{
        TensorDims{
            FFOrdered{
                40_p,
                48_p,
                36_p,
            },
        },
        DataType::FLOAT,
    };

    Tensorshape input_k = input_q;
    Tensorshape input_v = input_q;

    TensorShape result =
        attention_get_input_bias_shape(attrs, input_q, input_k, input_v);

    TensorShape correct = 
      TensorShape{
        TensorDims{
            FFOrdered{
                32_p,
            },
        },
        DataType::FLOAT,
    };

    ASSERT(result == correct);
  }

  TEST_CASE("attention_get_output_shape") {
    MultiHeadAttentionAttrs attrs = MultiHeadAttentionAttrs{
        /*embed_dim=*/32_p,
        /*num_heads=*/10_p,
        /*kdim=*/32_p,
        /*vdim=*/32_p,
        /*dropout=*/0.0,
        /*bias=*/true,
        /*add_bias_kv=*/false,
        /*add_zero_attn=*/false,
    };

    TensorShape input_q = TensorShape{
        TensorDims{
            FFOrdered{
                40_p,
                48_p,
                36_p,
            },
        },
        DataType::FLOAT,
    };

    Tensorshape input_k = input_q;
    Tensorshape input_v = input_q;

    TensorShape result =
        attention_get_output_shape(attrs, input_q, input_k, input_v);

    TensorShape correct = 
        TensorDims{
            FFOrdered{
                40_p,
                48_p,
                attrs.embed_dim,
            },
        },
        DataType::FLOAT,
    };
    CHECK(result == correct);
  }

  TEST_CASE("attention_get_weights_parallel_dim_degrees") {
    MultiHeadAttentionAttrs attrs = MultiHeadAttentionAttrs{
        /*embed_dim=*/32_p,
        /*num_heads=*/10_p,
        /*kdim=*/32_p,
        /*vdim=*/32_p,
        /*dropout=*/0.0,
        /*bias=*/true,
        /*add_bias_kv=*/false,
        /*add_zero_attn=*/false,
    };

    SUBCASE("data parallelism") {
      ParallelTensorDimDegrees q = mk_3d_dim_degrees(1, 1, 4, 1, 1);
      ParallelTensorDimDegrees k = q;
      ParallelTensorDimDegrees v = q;

      ParallelTensorDimDegrees result =
          attention_get_weights_parallel_dim_degrees(attrs, q, k, v);
      ParallelTensorDimDegrees correct = mk_2d_dim_degrees(1, 4, 1, 1);

      CHECK(result == correct);
    }

    SUBCASE("attention head parallelism") {
      ParallelTensorDimDegrees q = mk_3d_dim_degrees(1, 2, 1, 1, 1);
      ParallelTensorDimDegrees k = q;
      ParallelTensorDimDegrees v = q;

      ParallelTensorDimDegrees result =
          attention_get_weights_parallel_shape(attrs, q, k, v);
      ParallelTensorDimDegrees correct = mk_2d_dim_degrees(1, 1, 1, 2);

      CHECK(result == correct);
    }

    SUBCASE("combined data & attention head parallelism") {
      ParallelTensorDimDegrees q = mk_3d_dim_degrees(1, 2, 4, 1, 1);
      ParallelTensorDimDegrees k = q;
      ParallelTensorDimDegrees v = q;

      ParallelTensorDimDegrees result =
          attention_get_weights_parallel_shape(attrs, q, k, v);
      ParallelTensorDimDegrees correct = mk_2d_dim_degrees(1, 4, 1, 2);

      CHECK(result == correct);
    }
  }

  TEST_CASE("attention_get_input_bias_parallel_dim_degrees") {
    MultiHeadAttentionAttrs attrs = MultiHeadAttentionAttrs{
        /*embed_dim=*/32_p,
        /*num_heads=*/10_p,
        /*kdim=*/32_p,
        /*vdim=*/32_p,
        /*dropout=*/0.0,
        /*bias=*/true,
        /*add_bias_kv=*/false,
        /*add_zero_attn=*/false,
    };

    SUBCASE("data parallelism") {
      ParallelTensorDimDegrees q = mk_3d_dim_degrees(1, 1, 4, 1, 1);
      ParallelTensorDimDegrees k = q;
      ParallelTensorDimDegrees v = q;

      ParallelTensorDimDegrees result =
          attention_get_input_bias_parallel_dim_degrees(attrs, q, k, v);
      ParallelTensorDimDegrees correct = mk_1d_dim_degrees(1, 4, 1);

      CHECK(result == correct);
    }

    SUBCASE("attention head parallelism") {
      ParallelTensorDimDegrees q = mk_3d_dim_degrees(1, 2, 1, 1, 1);
      ParallelTensorDimDegrees k = q;
      ParallelTensorDimDegrees v = q;

      ParallelTensorDimDegrees result =
          attention_get_input_bias_parallel_shape(attrs, q, k, v);
      ParallelTensorDimDegrees correct = mk_1d_dim_degrees(1, 2, 1);

      CHECK(result == correct);
    }

    SUBCASE("combined data & attention head parallelism") {
      ParallelTensorDimDegrees q = mk_3d_dim_degrees(1, 2, 4, 1, 1);
      ParallelTensorDimDegrees k = q;
      ParallelTensorDimDegrees v = q;

      ParallelTensorDimDegrees result =
          attention_get_input_bias_parallel_shape(attrs, q, k, v);
      ParallelTensorDimDegrees correct = mk_1d_dim_degrees(1, 2*4, 1);

      CHECK(result == correct);
    }
  }

  TEST_CASE("attention_get_output_bias_parallel_dim_degrees") {
    MultiHeadAttentionAttrs attrs = MultiHeadAttentionAttrs{
        /*embed_dim=*/32_p,
        /*num_heads=*/10_p,
        /*kdim=*/32_p,
        /*vdim=*/32_p,
        /*dropout=*/0.0,
        /*bias=*/true,
        /*add_bias_kv=*/false,
        /*add_zero_attn=*/false,
    };

    SUBCASE("data parallelism") {
      ParallelTensorDimDegrees q = mk_3d_dim_degrees(1, 1, 4, 1, 1);
      ParallelTensorDimDegrees k = q;
      ParallelTensorDimDegrees v = q;

      ParallelTensorDimDegrees result =
          attention_get_output_bias_parallel_dim_degrees(attrs, q, k, v);
      ParallelTensorDimDegrees correct = mk_1d_dim_degrees(1, 4, 1);

      CHECK(result == correct);
    }

    SUBCASE("attention head parallelism") {
      ParallelTensorDimDegrees q = mk_3d_dim_degrees(1, 2, 1, 1, 1);
      ParallelTensorDimDegrees k = q;
      ParallelTensorDimDegrees v = q;

      ParallelTensorDimDegrees result =
          attention_get_output_bias_parallel_shape(attrs, q, k, v);
      ParallelTensorDimDegrees correct = mk_1d_dim_degrees(1, 2, 1);

      CHECK(result == correct);
    }

    SUBCASE("combined data & attention head parallelism") {
      ParallelTensorDimDegrees q = mk_3d_dim_degrees(1, 2, 4, 1, 1);
      ParallelTensorDimDegrees k = q;
      ParallelTensorDimDegrees v = q;

      ParallelTensorDimDegrees result =
          attention_get_output_bias_parallel_shape(attrs, q, k, v);
      ParallelTensorDimDegrees correct = mk_1d_dim_degrees(1, 2*4, 1);

      CHECK(result == correct);
    }
  }

  TEST_CASE("attention_get_output_parallel_dim_degrees") {
    MultiHeadAttentionAttrs attrs = MultiHeadAttentionAttrs{
        /*embed_dim=*/32_p,
        /*num_heads=*/10_p,
        /*kdim=*/32_p,
        /*vdim=*/32_p,
        /*dropout=*/0.0,
        /*bias=*/true,
        /*add_bias_kv=*/false,
        /*add_zero_attn=*/false,
    };

    SUBCASE("data parallelism") {
      ParallelTensorDimDegrees q = mk_3d_dim_degrees(1, 1, 4, 1, 1);
      ParallelTensorDimDegrees k = q;
      ParallelTensorDimDegrees v = q;

      ParallelTensorDimDegrees result =
          attention_get_output_parallel_dim_degrees(attrs, q, k, v);
      ParallelTensorDimDegrees correct = mk_3d_dim_degrees(1, 1, 4, 1, 1);

      CHECK(result == correct);
    }

    SUBCASE("attention head parallelism") {
      ParallelTensorDimDegrees q = mk_3d_dim_degrees(1, 2, 1, 1, 1);
      ParallelTensorDimDegrees k = q;
      ParallelTensorDimDegrees v = q;

      ParallelTensorDimDegrees result =
          attention_get_output_parallel_shape(attrs, q, k, v);
      ParallelTensorDimDegrees correct = mk_3d_dim_degrees(2, 1, 1, 1, 1);

      CHECK(result == correct);
    }

    SUBCASE("combined data & attention head parallelism") {
      ParallelTensorDimDegrees q = mk_3d_dim_degrees(1, 2, 4, 1, 1);
      ParallelTensorDimDegrees k = q;
      ParallelTensorDimDegrees v = q;

      ParallelTensorDimDegrees result =
          attention_get_output_parallel_shape(attrs, q, k, v);
      ParallelTensorDimDegrees correct = mk_3d_dim_degrees(2, 1, 4, 1, 1);

      CHECK(result == correct);
    }
  }

  TEST_CASE("attention_get_operator_to_parallel_tensor_mappings") {
    MultiHeadAttentionAttrs attrs = MultiHeadAttentionAttrs{
        /*embed_dim=*/32_p,
        /*num_heads=*/10_p,
        /*kdim=*/32_p,
        /*vdim=*/32_p,
        /*dropout=*/0.0,
        /*bias=*/true,
        /*add_bias_kv=*/false,
        /*add_zero_attn=*/false,
    };

    ParallelTensorDimDegrees q = mk_3d_dim_degrees(1, 2, 4, 1, 1);
    ParallelTensorDimDegrees k = q;
    ParallelTensorDimDegrees v = q;

    attention_get_operator_to_parallel_tensor_mappings(attrs, q, k, v);
    
    // for now just check that it doesn't crash
  }
}
