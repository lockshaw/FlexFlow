#include "op-attrs/ops/layer_norm.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/expected.h"
#include "utils/fmt/expected.h"
#include "utils/fmt/optional.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

static LayerNormAttrs mk_attrs(bool elementwise_affine) {
  return LayerNormAttrs{
      /*axes=*/{ff_dim_t{1_n}, ff_dim_t{3_n}},
      /*elementwise_affine=*/elementwise_affine,
      /*eps=*/0.1,
  };
}

template <typename... Args>
static ParallelTensorDimDegrees
    mk_degrees(int o_sum, int o_discard, Args... o_shards) {
  return ParallelTensorDimDegrees{
      SumDegree{positive_int{o_sum}},
      DiscardCopyDegree{positive_int{o_discard}},
      FFOrdered<positive_int>{positive_int{o_shards}...},
  };
}

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_layer_norm_incoming_tensor_roles") {
    auto make_attrs = [](bool elementwise_affine) {
      return LayerNormAttrs{
          /*axes=*/{ff_dim_t{0_n}, ff_dim_t{2_n}},
          elementwise_affine,
          /*eps=*/1.0,
      };
    };

    SUBCASE("elementwise_affine = true") {
      LayerNormAttrs attrs = make_attrs(/*elementwise_affine=*/true);

      std::map<TensorSlotName, IncomingTensorRole> result =
          get_layer_norm_incoming_tensor_roles(attrs);
      std::map<TensorSlotName, IncomingTensorRole> correct = {
          {
              TensorSlotName::INPUT,
              IncomingTensorRole::INPUT,
          },
          {
              TensorSlotName::GAMMA,
              IncomingTensorRole::WEIGHT,
          },
          {
              TensorSlotName::BETA,
              IncomingTensorRole::WEIGHT,
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("elementwise_affine = false") {
      LayerNormAttrs attrs = make_attrs(/*elementwise_affine=*/false);

      std::map<TensorSlotName, IncomingTensorRole> result =
          get_layer_norm_incoming_tensor_roles(attrs);
      std::map<TensorSlotName, IncomingTensorRole> correct = {
          {
              TensorSlotName::INPUT,
              IncomingTensorRole::INPUT,
          },
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("layer_norm_get_output_shape") {
    TensorShape input = TensorShape{
        TensorDims{
            FFOrdered<positive_int>{
                12_p,
                8_p,
                13_p,
                10_p,
            },
        },
        DataType::FLOAT,
    };

    SUBCASE("elementwise_affine = true") {
      LayerNormAttrs attrs = mk_attrs(true);

      TensorShape result = layer_norm_get_output_shape(attrs, input);
      TensorShape correct = input;

      CHECK(result == correct);
    }

    SUBCASE("elementwise_affine = false") {
      LayerNormAttrs attrs = mk_attrs(false);

      TensorShape result = layer_norm_get_output_shape(attrs, input);
      TensorShape correct = input;

      CHECK(result == correct);
    }
  }

  TEST_CASE("layer_norm_get_gamma_weights_shape") {
    TensorShape input = TensorShape{
        TensorDims{
            FFOrdered<positive_int>{
                12_p,
                8_p,
                13_p,
                10_p,
            },
        },
        DataType::FLOAT,
    };

    SUBCASE("elementwise_affine = true") {
      LayerNormAttrs attrs = mk_attrs(true);

      TensorShape result = layer_norm_get_gamma_weights_shape(attrs, input);
      TensorShape correct = TensorShape{
          TensorDims{
              FFOrdered<positive_int>{
                  8_p,
                  10_p,
              },
          },
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }

    SUBCASE("elementwise_affine = false") {
      LayerNormAttrs attrs = mk_attrs(false);

      CHECK_THROWS(layer_norm_get_gamma_weights_shape(attrs, input));
    }
  }

  TEST_CASE("layer_norm_get_beta_weights_shape") {
    TensorShape input = TensorShape{
        TensorDims{
            FFOrdered<positive_int>{
                12_p,
                8_p,
                13_p,
                10_p,
            },
        },
        DataType::FLOAT,
    };

    SUBCASE("elementwise_affine = true") {
      LayerNormAttrs attrs = mk_attrs(true);

      TensorShape result = layer_norm_get_beta_weights_shape(attrs, input);
      TensorShape correct = TensorShape{
          TensorDims{
              FFOrdered<positive_int>{
                  8_p,
                  10_p,
              },
          },
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }

    SUBCASE("elementwise_affine = false") {
      LayerNormAttrs attrs = mk_attrs(false);

      CHECK_THROWS(layer_norm_get_beta_weights_shape(attrs, input));
    }
  }

  TEST_CASE("layer_norm_get_output_parallel_dim_degrees") {
    LayerNormAttrs affine_true_attrs = mk_attrs(true);
    LayerNormAttrs affine_false_attrs = mk_attrs(false);

    SUBCASE("preexisting sum parallelism") {
      ParallelTensorDimDegrees input = mk_degrees(3, 1, 1, 1, 1, 1);

      SUBCASE("elementwise_affine = true") {
        CHECK_THROWS(layer_norm_get_output_parallel_dim_degrees(
            affine_true_attrs, input));
      }

      SUBCASE("elementwise_affine = false") {
        CHECK_THROWS(layer_norm_get_output_parallel_dim_degrees(
            affine_false_attrs, input));
      }
    }

    SUBCASE("preexisting discard copy parallelism") {
      ParallelTensorDimDegrees input = mk_degrees(1, 3, 1, 1, 1, 1);

      SUBCASE("elementwise_affine = true") {
        ParallelTensorDimDegrees result =
            layer_norm_get_output_parallel_dim_degrees(affine_true_attrs,
                                                       input);

        ParallelTensorDimDegrees correct = mk_degrees(3, 1, 1, 1, 1, 1);

        CHECK(result == correct);
      }

      SUBCASE("elementwise_affine = false") {
        ParallelTensorDimDegrees result =
            layer_norm_get_output_parallel_dim_degrees(affine_false_attrs,
                                                       input);

        ParallelTensorDimDegrees correct = mk_degrees(3, 1, 1, 1, 1, 1);

        CHECK(result == correct);
      }
    }

    SUBCASE("shard parallelism in norm axes") {
      ParallelTensorDimDegrees input = mk_degrees(1, 1, 1, 2, 1, 1);

      SUBCASE("elementwise_affine = true") {
        CHECK_THROWS(layer_norm_get_output_parallel_dim_degrees(
            affine_true_attrs, input));
      }

      SUBCASE("elementwise_affine = false") {
        CHECK_THROWS(layer_norm_get_output_parallel_dim_degrees(
            affine_false_attrs, input));
      }
    }

    SUBCASE("shard parallelism not in norm axes") {
      ParallelTensorDimDegrees input = mk_degrees(1, 1, 2, 1, 1, 1);

      SUBCASE("elementwise_affine = true") {
        ParallelTensorDimDegrees result =
            layer_norm_get_output_parallel_dim_degrees(affine_true_attrs,
                                                       input);

        ParallelTensorDimDegrees correct = input;

        CHECK(result == correct);
      }

      SUBCASE("elementwise_affine = false") {
        ParallelTensorDimDegrees result =
            layer_norm_get_output_parallel_dim_degrees(affine_false_attrs,
                                                       input);

        ParallelTensorDimDegrees correct = input;

        CHECK(result == correct);
      }
    }

    SUBCASE("hybrid parallelism") {
      ParallelTensorDimDegrees input = mk_degrees(1, 5, 7, 1, 2, 1);

      SUBCASE("elementwise_affine = true") {
        ParallelTensorDimDegrees result =
            layer_norm_get_output_parallel_dim_degrees(affine_true_attrs,
                                                       input);

        ParallelTensorDimDegrees correct = mk_degrees(5, 1, 7, 1, 2, 1);

        CHECK(result == correct);
      }

      SUBCASE("elementwise_affine = false") {
        ParallelTensorDimDegrees result =
            layer_norm_get_output_parallel_dim_degrees(affine_false_attrs,
                                                       input);

        ParallelTensorDimDegrees correct = mk_degrees(5, 1, 7, 1, 2, 1);

        CHECK(result == correct);
      }
    }
  }

  TEST_CASE("layer_norm_get_gamma_weights_parallel_dim_degrees") {
    LayerNormAttrs affine_true_attrs = mk_attrs(true);
    LayerNormAttrs affine_false_attrs = mk_attrs(false);

    SUBCASE("preexisting sum parallelism") {
      ParallelTensorDimDegrees input = mk_degrees(3, 1, 1, 1, 1, 1);

      SUBCASE("elementwise_affine = true") {
        CHECK_THROWS(layer_norm_get_gamma_weights_parallel_dim_degrees(
            affine_true_attrs, input));
      }

      SUBCASE("elementwise_affine = false") {
        CHECK_THROWS(layer_norm_get_gamma_weights_parallel_dim_degrees(
            affine_false_attrs, input));
      }
    }

    SUBCASE("preexisting discard copy parallelism") {
      ParallelTensorDimDegrees input = mk_degrees(1, 3, 1, 1, 1, 1);

      SUBCASE("elementwise_affine = true") {
        ParallelTensorDimDegrees result =
            layer_norm_get_gamma_weights_parallel_dim_degrees(affine_true_attrs,
                                                              input);

        ParallelTensorDimDegrees correct = mk_degrees(3, 1, 1, 1);

        CHECK(result == correct);
      }

      SUBCASE("elementwise_affine = false") {
        CHECK_THROWS(layer_norm_get_gamma_weights_parallel_dim_degrees(
            affine_false_attrs, input));
      }
    }

    SUBCASE("shard parallelism in norm axes") {
      ParallelTensorDimDegrees input = mk_degrees(1, 1, 1, 2, 1, 1);

      SUBCASE("elementwise_affine = true") {
        CHECK_THROWS(layer_norm_get_gamma_weights_parallel_dim_degrees(
            affine_true_attrs, input));
      }

      SUBCASE("elementwise_affine = false") {
        CHECK_THROWS(layer_norm_get_gamma_weights_parallel_dim_degrees(
            affine_false_attrs, input));
      }
    }

    SUBCASE("shard parallelism not in norm axes") {
      ParallelTensorDimDegrees input = mk_degrees(1, 1, 2, 1, 1, 1);

      SUBCASE("elementwise_affine = true") {
        ParallelTensorDimDegrees result =
            layer_norm_get_gamma_weights_parallel_dim_degrees(affine_true_attrs,
                                                              input);

        ParallelTensorDimDegrees correct = mk_degrees(1, 2, 1, 1);

        CHECK(result == correct);
      }

      SUBCASE("elementwise_affine = false") {
        CHECK_THROWS(layer_norm_get_gamma_weights_parallel_dim_degrees(
            affine_false_attrs, input));
      }
    }

    SUBCASE("hybrid parallelism") {
      ParallelTensorDimDegrees input = mk_degrees(1, 5, 7, 1, 2, 1);

      SUBCASE("elementwise_affine = true") {
        ParallelTensorDimDegrees result =
            layer_norm_get_gamma_weights_parallel_dim_degrees(affine_true_attrs,
                                                              input);

        ParallelTensorDimDegrees correct = mk_degrees(5, 7 * 2, 1, 1);

        CHECK(result == correct);
      }

      SUBCASE("elementwise_affine = false") {
        CHECK_THROWS(layer_norm_get_gamma_weights_parallel_dim_degrees(
            affine_false_attrs, input));
      }
    }
  }

  TEST_CASE("layer_norm_get_beta_weights_parallel_dim_degrees") {
    LayerNormAttrs affine_true_attrs = mk_attrs(true);
    LayerNormAttrs affine_false_attrs = mk_attrs(false);

    SUBCASE("preexisting sum parallelism") {
      ParallelTensorDimDegrees input = mk_degrees(3, 1, 1, 1, 1, 1);

      SUBCASE("elementwise_affine = true") {
        CHECK_THROWS(layer_norm_get_beta_weights_parallel_dim_degrees(
            affine_true_attrs, input));
      }

      SUBCASE("elementwise_affine = false") {
        CHECK_THROWS(layer_norm_get_beta_weights_parallel_dim_degrees(
            affine_false_attrs, input));
      }
    }

    SUBCASE("preexisting discard copy parallelism") {
      ParallelTensorDimDegrees input = mk_degrees(1, 3, 1, 1, 1, 1);

      SUBCASE("elementwise_affine = true") {
        ParallelTensorDimDegrees result =
            layer_norm_get_beta_weights_parallel_dim_degrees(affine_true_attrs,
                                                             input);

        ParallelTensorDimDegrees correct = mk_degrees(3, 1, 1, 1);

        CHECK(result == correct);
      }

      SUBCASE("elementwise_affine = false") {
        CHECK_THROWS(layer_norm_get_beta_weights_parallel_dim_degrees(
            affine_false_attrs, input));
      }
    }

    SUBCASE("shard parallelism in norm axes") {
      ParallelTensorDimDegrees input = mk_degrees(1, 1, 1, 2, 1, 1);

      SUBCASE("elementwise_affine = true") {
        CHECK_THROWS(layer_norm_get_beta_weights_parallel_dim_degrees(
            affine_true_attrs, input));
      }

      SUBCASE("elementwise_affine = false") {
        CHECK_THROWS(layer_norm_get_beta_weights_parallel_dim_degrees(
            affine_false_attrs, input));
      }
    }

    SUBCASE("shard parallelism not in norm axes") {
      ParallelTensorDimDegrees input = mk_degrees(1, 1, 2, 1, 1, 1);

      SUBCASE("elementwise_affine = true") {
        ParallelTensorDimDegrees result =
            layer_norm_get_beta_weights_parallel_dim_degrees(affine_true_attrs,
                                                             input);

        ParallelTensorDimDegrees correct = mk_degrees(1, 2, 1, 1);

        CHECK(result == correct);
      }

      SUBCASE("elementwise_affine = false") {
        CHECK_THROWS(layer_norm_get_beta_weights_parallel_dim_degrees(
            affine_false_attrs, input));
      }
    }

    SUBCASE("hybrid parallelism") {
      ParallelTensorDimDegrees input = mk_degrees(1, 5, 7, 1, 2, 1);

      SUBCASE("elementwise_affine = true") {
        ParallelTensorDimDegrees result =
            layer_norm_get_beta_weights_parallel_dim_degrees(affine_true_attrs,
                                                             input);

        ParallelTensorDimDegrees correct = mk_degrees(5, 7 * 2, 1, 1);

        CHECK(result == correct);
      }

      SUBCASE("elementwise_affine = false") {
        CHECK_THROWS(layer_norm_get_beta_weights_parallel_dim_degrees(
            affine_false_attrs, input));
      }
    }
  }

  TEST_CASE("layer_norm_get_output_parallel_shape") {
    // since most of the edge cases are already tested in
    // layer_norm_get_output_shape and
    // layer_norm_get_output_parallel_dim_degrees, here we just do a
    // basic check that they compose

    ParallelTensorShape input_shape = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{49_p, 7_p},
                ShardParallelDim{13_p, 1_p},
                ShardParallelDim{16_p, 2_p},
                ShardParallelDim{12_p, 1_p},
            },
            ReplicaParallelDimSet{
                SumDegree{1_p},
                DiscardCopyDegree{5_p},
            },
        },
        DataType::FLOAT,
    };

    SUBCASE("elementwise_affine = true") {
      LayerNormAttrs attrs = mk_attrs(true);

      ParallelTensorShape result =
          layer_norm_get_output_parallel_shape(attrs, input_shape);

      ParallelTensorShape correct = ParallelTensorShape{
          ParallelTensorDims{
              FFOrdered<ShardParallelDim>{
                  ShardParallelDim{49_p, 7_p},
                  ShardParallelDim{13_p, 1_p},
                  ShardParallelDim{16_p, 2_p},
                  ShardParallelDim{12_p, 1_p},
              },
              ReplicaParallelDimSet{
                  SumDegree{5_p},
                  DiscardCopyDegree{1_p},
              },
          },
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }

    SUBCASE("elementwise_affine = false") {
      LayerNormAttrs attrs = mk_attrs(false);

      ParallelTensorShape result =
          layer_norm_get_output_parallel_shape(attrs, input_shape);

      ParallelTensorShape correct = ParallelTensorShape{
          ParallelTensorDims{
              FFOrdered<ShardParallelDim>{
                  ShardParallelDim{49_p, 7_p},
                  ShardParallelDim{13_p, 1_p},
                  ShardParallelDim{16_p, 2_p},
                  ShardParallelDim{12_p, 1_p},
              },
              ReplicaParallelDimSet{
                  SumDegree{5_p},
                  DiscardCopyDegree{1_p},
              },
          },
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("layer_norm_get_gamma_weights_parallel_shape") {
    // since most of the edge cases are already tested in
    // layer_norm_get_gamma_weights_shape and
    // layer_norm_get_gamma_weights_parallel_dim_degrees, here we just do a
    // basic check that they compose

    ParallelTensorShape input_shape = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{49_p, 7_p},
                ShardParallelDim{13_p, 1_p},
                ShardParallelDim{16_p, 2_p},
                ShardParallelDim{12_p, 1_p},
            },
            ReplicaParallelDimSet{
                SumDegree{1_p},
                DiscardCopyDegree{5_p},
            },
        },
        DataType::FLOAT,
    };

    SUBCASE("elementwise_affine = true") {
      LayerNormAttrs attrs = mk_attrs(true);

      ParallelTensorShape result =
          layer_norm_get_gamma_weights_parallel_shape(attrs, input_shape);

      ParallelTensorShape correct = ParallelTensorShape{
          ParallelTensorDims{
              FFOrdered<ShardParallelDim>{
                  ShardParallelDim{13_p, 1_p},
                  ShardParallelDim{12_p, 1_p},
              },
              ReplicaParallelDimSet{
                  SumDegree{5_p},
                  DiscardCopyDegree{7_p * 2_p},
              },
          },
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }

    SUBCASE("elementwise_affine = false") {
      LayerNormAttrs attrs = mk_attrs(false);

      CHECK_THROWS(
          layer_norm_get_gamma_weights_parallel_shape(attrs, input_shape));
    }
  }

  TEST_CASE("layer_norm_get_beta_weights_parallel_shape") {
    // since most of the edge cases are already tested in
    // layer_norm_get_beta_weights_shape and
    // layer_norm_get_beta_weights_parallel_dim_degrees, here we just do a
    // basic check that they compose

    ParallelTensorShape input_shape = ParallelTensorShape{
        ParallelTensorDims{
            FFOrdered<ShardParallelDim>{
                ShardParallelDim{49_p, 7_p},
                ShardParallelDim{13_p, 1_p},
                ShardParallelDim{16_p, 2_p},
                ShardParallelDim{12_p, 1_p},
            },
            ReplicaParallelDimSet{
                SumDegree{1_p},
                DiscardCopyDegree{5_p},
            },
        },
        DataType::FLOAT,
    };

    SUBCASE("elementwise_affine = true") {
      LayerNormAttrs attrs = mk_attrs(true);

      ParallelTensorShape result =
          layer_norm_get_beta_weights_parallel_shape(attrs, input_shape);

      ParallelTensorShape correct = ParallelTensorShape{
          ParallelTensorDims{
              FFOrdered<ShardParallelDim>{
                  ShardParallelDim{13_p, 1_p},
                  ShardParallelDim{12_p, 1_p},
              },
              ReplicaParallelDimSet{
                  SumDegree{5_p},
                  DiscardCopyDegree{7_p * 2_p},
              },
          },
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }

    SUBCASE("elementwise_affine = false") {
      LayerNormAttrs attrs = mk_attrs(false);

      CHECK_THROWS(
          layer_norm_get_beta_weights_parallel_shape(attrs, input_shape));
    }
  }
}
