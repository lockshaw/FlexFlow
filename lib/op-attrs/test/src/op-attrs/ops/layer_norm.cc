#include "op-attrs/ops/layer_norm.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/expected.h"
#include "utils/fmt/expected.h"
#include "utils/fmt/optional.h"
#include <doctest/doctest.h>
#include "kernels/local_cpu_allocator.h"
#include "kernels/accessor.h"
#include "kernels/create_zero_filled_accessor.h"
#include "kernels/layer_norm_kernels_cpu.h"
#include "kernels/shard_signature_instance_is_valid.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_layer_norm_incoming_tensor_roles(LayerNormAttrs)") {
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

  TEST_CASE("shape inference (LayerNorm)") {
    LayerNormAttrs attrs_affine_true = LayerNormAttrs{
        /*axes=*/{ff_dim_t{1_n}, ff_dim_t{3_n}},
        /*elementwise_affine=*/true,
        /*eps=*/0.1,
    };

    LayerNormAttrs attrs_affine_false = [&] {
      LayerNormAttrs attrs = attrs_affine_true;
      attrs.elementwise_affine = false;
      return attrs;
    }();

    TensorShape input = TensorShape{
        TensorDims{FFOrdered{
            12_p,
            14_p,
            16_p,
            18_p,
        }},
        DataType::FLOAT,
    };

    TensorShape output = input;

    TensorShape gamma = TensorShape{
        TensorDims{FFOrdered{
            14_p,
            18_p,
        }},
        DataType::FLOAT,
    };

    TensorShape beta = gamma;

    SUBCASE("layer_norm_get_output_shape(LayerNormAttrs, TensorShape)") {
      TensorShape result =
          layer_norm_get_output_shape(attrs_affine_true, input);
      TensorShape correct = output;

      CHECK(result == correct);
    }

    SUBCASE("layer_norm_get_gamma_weights_shape(LayerNormAttrs, TensorShape)") {
      SUBCASE("elementwise_affine = true") {
        TensorShape result =
            layer_norm_get_gamma_weights_shape(attrs_affine_true, input);
        TensorShape correct = gamma;

        CHECK(result == correct);
      }

      SUBCASE("elementwise_affine = false") {
        CHECK_THROWS(layer_norm_get_gamma_weights_shape(attrs_affine_false, input));
      }
    }

    SUBCASE("layer_norm_get_beta_weights_shape(LayerNormAttrs, TensorShape)") {
      SUBCASE("elementwise_affine = true") {
        TensorShape result =
            layer_norm_get_beta_weights_shape(attrs_affine_true, input);
        TensorShape correct = beta;

        CHECK(result == correct);
      }

      SUBCASE("elementwise_affine = false") {
        CHECK_THROWS(layer_norm_get_beta_weights_shape(attrs_affine_false, input));
      }
    }

    auto make_input = [&](SumDegree o_sum,
                          DiscardCopyDegree o_eq,
                          positive_int o0,
                          positive_int o1,
                          positive_int o2,
                          positive_int o3)
      -> ParallelTensorShape
    {
      return lift_shape_to_parallel_with_degrees(
          input, o_sum, o_eq, FFOrdered{o0, o1, o2, o3});
    };

    auto make_output = [&](SumDegree o_sum,
                           DiscardCopyDegree o_eq,
                           positive_int o0,
                           positive_int o1,
                           positive_int o2,
                           positive_int o3)
      -> ParallelTensorShape
    {
      return lift_shape_to_parallel_with_degrees(
          output, o_sum, o_eq, FFOrdered{o0, o1, o2, o3});
    };

    auto make_gamma_weights = [&](SumDegree o_sum,
                                  DiscardCopyDegree o_eq,
                                  positive_int o0,
                                  positive_int o2)
      -> ParallelTensorShape
    {
      return lift_shape_to_parallel_with_degrees(
          gamma, o_sum, o_eq, FFOrdered{o0, o2});
    };

    auto make_beta_weights = [&](SumDegree o_sum,
                                 DiscardCopyDegree o_eq,
                                 positive_int o0,
                                 positive_int o2)
      -> ParallelTensorShape
    {
      return lift_shape_to_parallel_with_degrees(
          beta, o_sum, o_eq, FFOrdered{o0, o2});
    };

    SUBCASE("parallel shape inference (LayerNorm)") {
      SUBCASE("partition parallelism (not in axes)") {
        positive_int degree0 = 2_p;
        positive_int degree2 = 3_p;

        ParallelTensorShape par_input = make_input(
            SumDegree{1_p}, DiscardCopyDegree{1_p}, degree0, 1_p, degree2, 1_p);

        SUBCASE("layer_norm_get_output_parallel_shape(LayerNormAttrs, ParallelTensorShape)") {
          ParallelTensorShape result =
              layer_norm_get_output_parallel_shape(attrs_affine_true, par_input);
          ParallelTensorShape correct =
              make_output(SumDegree{1_p},
                          DiscardCopyDegree{1_p},
                          degree0,
                          1_p,
                          degree2,
                          1_p);

          CHECK(result == correct);
        }

        SUBCASE(
            "layer_norm_get_gamma_weights_parallel_shape(LayerNormAttrs, ParallelTensorShape)") {
          SUBCASE("elementwise_affine = true") {
            ParallelTensorShape result =
                layer_norm_get_gamma_weights_parallel_shape(attrs_affine_true, par_input);
            ParallelTensorShape correct =
                make_gamma_weights(
                    SumDegree{1_p}, DiscardCopyDegree{6_p}, 1_p, 1_p);

            CHECK(result == correct);
          }

          SUBCASE("elementwise_affine = false") {
            CHECK_THROWS(layer_norm_get_gamma_weights_parallel_shape(attrs_affine_false, par_input));
          }
        }

        SUBCASE("layer_norm_get_beta_weights_parallel_shape(LayerNormAttrs, ParallelTensorShape)") {
          SUBCASE("elementwise_affine = true") {
            ParallelTensorShape result =
                layer_norm_get_beta_weights_parallel_shape(attrs_affine_true, par_input);
            ParallelTensorShape correct =
                make_beta_weights(
                    SumDegree{1_p}, DiscardCopyDegree{6_p}, 1_p, 1_p);

            CHECK(result == correct);
          }

          SUBCASE("elementwise_affine = false") {
            CHECK_THROWS(layer_norm_get_beta_weights_parallel_shape(attrs_affine_false, par_input));
          }
        }
      }

      SUBCASE("partition parallelism (in axes)") {
        positive_int degree1 = 2_p;
        positive_int degree2 = 4_p;

        ParallelTensorShape par_input = make_input(
            SumDegree{1_p}, DiscardCopyDegree{1_p}, 1_p, degree1, degree2, 1_p);

        SUBCASE("layer_norm_get_output_parallel_shape(LayerNormAttrs, ParallelTensorShape)") {
          CHECK_THROWS(layer_norm_get_output_parallel_shape(attrs_affine_true, par_input));
        }

        SUBCASE(
            "layer_norm_get_gamma_weights_parallel_shape(LayerNormAttrs, ParallelTensorShape)") {
          CHECK_THROWS(layer_norm_get_gamma_weights_parallel_shape(attrs_affine_true, par_input));
        }

        SUBCASE("layer_norm_get_beta_weights_parallel_shape(LayerNormAttrs, ParallelTensorShape)") {
          CHECK_THROWS(layer_norm_get_beta_weights_parallel_shape(attrs_affine_true, par_input));
        }
      }

      SUBCASE("sum parallelism") {
        SumDegree sum_degree = SumDegree{2_p};

        ParallelTensorShape par_input =
            make_input(sum_degree, DiscardCopyDegree{1_p}, 1_p, 1_p, 1_p, 1_p);

        SUBCASE("layer_norm_get_output_parallel_shape(LayerNormAttrs, ParallelTensorShape)") {
          CHECK_THROWS(layer_norm_get_output_parallel_shape(attrs_affine_true, par_input));
        }

        SUBCASE(
            "layer_norm_get_gamma_weights_parallel_shape(LayerNormAttrs, ParallelTensorShape)") {
          CHECK_THROWS(layer_norm_get_gamma_weights_parallel_shape(attrs_affine_true, par_input));
        }

        SUBCASE("layer_norm_get_beta_weights_parallel_shape(LayerNormAttrs, ParallelTensorShape)") {
          CHECK_THROWS(layer_norm_get_beta_weights_parallel_shape(attrs_affine_true, par_input));
        }
      }

      SUBCASE("discard copy parallelism") {
        DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{2_p};

        ParallelTensorShape par_input =
            make_input(SumDegree{1_p}, discard_copy_degree, 1_p, 1_p, 1_p, 1_p);

        SUBCASE("layer_norm_get_output_parallel_shape(LayerNormAttrs, ParallelTensorShape)") {
          ParallelTensorShape result =
              layer_norm_get_output_parallel_shape(attrs_affine_true, par_input);
          ParallelTensorShape correct =
              make_output(SumDegree{2_p}, DiscardCopyDegree{1_p}, 1_p, 1_p, 1_p, 1_p);

          CHECK(result == correct);
        }

        SUBCASE(
            "layer_norm_get_gamma_weights_parallel_shape(LayerNormAttrs, ParallelTensorShape)") {
            ParallelTensorShape result =
                layer_norm_get_gamma_weights_parallel_shape(attrs_affine_true, par_input);
            ParallelTensorShape correct =
                make_gamma_weights(
                    SumDegree{2_p}, DiscardCopyDegree{1_p}, 1_p, 1_p);

            CHECK(result == correct);
        }

        SUBCASE("layer_norm_get_beta_weights_parallel_shape(LayerNormAttrs, ParallelTensorShape)") {
            ParallelTensorShape result =
                layer_norm_get_beta_weights_parallel_shape(attrs_affine_true, par_input);
            ParallelTensorShape correct =
                make_beta_weights(
                    SumDegree{2_p}, DiscardCopyDegree{1_p}, 1_p, 1_p);

            CHECK(result == correct);
        }
      }
    }
  }

  TEST_CASE("layer_norm_get_shard_signature_instance") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    TensorShape input_shape = TensorShape{
      TensorDims{
        FFOrdered{
          6_p,
          4_p,
          5_p,
        },
      },
      DataType::FLOAT,
    };

    auto mk_dim_degrees = [&](int sum_degree,
                              int discard_copy_degree,
                              int dim0_shard_degree,
                              int dim1_shard_degree,
                              int dim2_shard_degree)
      -> ParallelTensorDimDegrees
    {
      return ParallelTensorDimDegrees{
        /*sum_degree=*/SumDegree{positive_int{sum_degree}},
        /*discard_copy_degree=*/DiscardCopyDegree{positive_int{discard_copy_degree}},
        /*shard_degrees=*/FFOrdered{
          positive_int{dim0_shard_degree},
          positive_int{dim1_shard_degree},
          positive_int{dim2_shard_degree},
        },
      };
    };

    auto run_layer_norm = [&](LayerNormAttrs const &attrs, 
                              std::map<TensorSlotName, GenericTensorAccessorR> const &incoming_shards)
      -> std::map<TensorSlotName, GenericTensorAccessorR>
    {
      GenericTensorAccessorR input_shard = incoming_shards.at(TensorSlotName::INPUT);
      TensorShape output_shard_shape =
        layer_norm_get_output_shape(attrs, get_tensor_shape_for_accessor_r(input_shard));
      GenericTensorAccessorW output_shard = create_zero_filled_accessor_w(output_shard_shape, cpu_allocator);

      layer_norm_cpu_forward_kernel(
        /*attrs=*/attrs,
        /*input=*/input_shard,
        /*output=*/output_shard,
        /*gamma=*/try_at(incoming_shards, TensorSlotName::GAMMA),
        /*beta=*/try_at(incoming_shards, TensorSlotName::BETA));

      return std::map<TensorSlotName, GenericTensorAccessorR>{
        {
          TensorSlotName::OUTPUT,
          read_only_accessor_from_write_accessor(output_shard),
        },
      };
    };

    auto layer_norm_shard_signature_instance_is_valid = [&](LayerNormAttrs const &attrs,
                                                            ParallelTensorDimDegrees const &input_degrees)
      -> bool
    {
      ParallelTensorShape input_parallel_shape = lift_shape_to_parallel_with_degrees(input_shape, input_degrees);

      std::map<TensorSlotName, ParallelTensorShape> input_shapes = {
        {
          TensorSlotName::INPUT,
          input_parallel_shape,
        },
      };

      return shard_signature_instance_is_valid(
        /*attrs=*/ComputationGraphOpAttrs{attrs},
        /*input_shapes=*/input_shapes,
        /*run_op=*/
          [&](std::map<TensorSlotName, GenericTensorAccessorR> const &incoming_shards) 
            -> std::map<TensorSlotName, GenericTensorAccessorR>
          {
            return run_layer_norm(attrs, incoming_shards);
          },
        /*seed=*/0);
    };

    SUBCASE("elementwise_affine = true") {
      LayerNormAttrs attrs = LayerNormAttrs{
        /*axes=*/std::set<ff_dim_t>{
          ff_dim_t{1_n}, 
          ff_dim_t{2_n},
        },
        /*elementwise_affine=*/true,
        /*eps=*/1.0f,
      };

      SUBCASE("data parallelism") {
        ParallelTensorDimDegrees input_dim_degrees = mk_dim_degrees(1, 1, 2, 1, 1);

        CHECK(layer_norm_shard_signature_instance_is_valid(attrs, input_dim_degrees));
      }
    }

    SUBCASE("elementwise_affine = false") {
      LayerNormAttrs attrs = LayerNormAttrs{
        /*axes=*/std::set<ff_dim_t>{
          ff_dim_t{1_n}, 
          ff_dim_t{2_n},
        },
        /*elementwise_affine=*/true,
        /*eps=*/1.0f,
      };

      SUBCASE("data parallelism") {
        ParallelTensorDimDegrees input_dim_degrees = mk_dim_degrees(1, 1, 2, 1, 1);

        CHECK(layer_norm_shard_signature_instance_is_valid(attrs, input_dim_degrees));
      }
    }
  }
}
