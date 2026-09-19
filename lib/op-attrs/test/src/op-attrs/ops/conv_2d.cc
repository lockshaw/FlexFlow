#include "op-attrs/ops/conv_2d.h"
#include "utils/integer_conversions.h"
#include <doctest/doctest.h>
#include "utils/orthotope/orthotope_bounded_coord.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_conv2d_incoming_tensor_roles") {
    auto make_attrs = [](bool use_bias) {
      return Conv2DAttrs{/*out_channels=*/4_p,
                         /*kernel_h=*/3_p,
                         /*kernel_w=*/2_p,
                         /*stride_h=*/2_p,
                         /*stride_w=*/2_p,
                         /*padding_h=*/1_n,
                         /*padding_w=*/1_n,
                         /*groups=*/1_p,
                         /*activation=*/std::nullopt,
                         /*use_bias=*/use_bias};
    };

    SUBCASE("with bias") {
      Conv2DAttrs attrs = make_attrs(/*use_bias=*/true);

      std::map<TensorSlotName, IncomingTensorRole> result =
          get_conv2d_incoming_tensor_roles(attrs);
      std::map<TensorSlotName, IncomingTensorRole> correct = {
          {
              TensorSlotName::INPUT,
              IncomingTensorRole::INPUT,
          },
          {
              TensorSlotName::FILTER,
              IncomingTensorRole::WEIGHT,
          },
          {
              TensorSlotName::BIAS,
              IncomingTensorRole::WEIGHT,
          },
      };

      CHECK(result == correct);
    }

    SUBCASE("without bias") {
      Conv2DAttrs attrs = make_attrs(/*use_bias=*/false);

      std::map<TensorSlotName, IncomingTensorRole> result =
          get_conv2d_incoming_tensor_roles(attrs);
      std::map<TensorSlotName, IncomingTensorRole> correct = {
          {
              TensorSlotName::INPUT,
              IncomingTensorRole::INPUT,
          },
          {
              TensorSlotName::FILTER,
              IncomingTensorRole::WEIGHT,
          },
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("conv2d_get_kernel_shape") {
    TensorShape input = TensorShape{
        TensorDims{FFOrdered{
            7_p,
            4_p,
            11_p,
            15_p,
        }},
        DataType::FLOAT,
    };

    SUBCASE("groups = 1") {
      Conv2DAttrs attrs = Conv2DAttrs{
          /*out_channels=*/13_p,
          /*kernel_h=*/3_p,
          /*kernel_w=*/4_p,
          /*stride_h=*/1_p,
          /*stride_w=*/1_p,
          /*padding_h=*/0_n,
          /*padding_w=*/0_n,
          /*groups=*/1_p,
          /*activation=*/std::nullopt,
          /*use_bias=*/true,
      };

      TensorShape result = conv2d_get_kernel_shape(attrs, input);

      TensorShape correct = TensorShape{
          TensorDims{FFOrdered{
              13_p,
              4_p,
              3_p,
              4_p,
          }},
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }

    SUBCASE("groups > 1") {
      Conv2DAttrs attrs = Conv2DAttrs{
          /*out_channels=*/13_p,
          /*kernel_h=*/3_p,
          /*kernel_w=*/4_p,
          /*stride_h=*/1_p,
          /*stride_w=*/1_p,
          /*padding_h=*/0_n,
          /*padding_w=*/0_n,
          /*groups=*/2_p,
          /*activation=*/std::nullopt,
          /*use_bias=*/true,
      };

      TensorShape result = conv2d_get_kernel_shape(attrs, input);

      TensorShape correct = TensorShape{
          TensorDims{FFOrdered{
              13_p,
              2_p,
              3_p,
              4_p,
          }},
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("conv2d_get_bias_shape") {
    TensorShape input = TensorShape{
        TensorDims{FFOrdered{
            7_p,
            4_p,
            11_p,
            15_p,
        }},
        DataType::FLOAT,
    };

    Conv2DAttrs attrs = Conv2DAttrs{
        /*out_channels=*/13_p,
        /*kernel_h=*/3_p,
        /*kernel_w=*/4_p,
        /*stride_h=*/1_p,
        /*stride_w=*/1_p,
        /*padding_h=*/0_n,
        /*padding_w=*/0_n,
        /*groups=*/1_p,
        /*activation=*/std::nullopt,
        /*use_bias=*/true,
    };

    TensorShape result = conv2d_get_bias_shape(attrs, input);

    TensorShape correct = TensorShape{
        TensorDims{FFOrdered{
            13_p,
        }},
        DataType::FLOAT,
    };

    CHECK(result == correct);
  }

  TEST_CASE("conv2d_get_output_shape") {
    TensorShape input = TensorShape{
        TensorDims{FFOrdered{
            7_p,
            4_p,
            11_p,
            15_p,
        }},
        DataType::FLOAT,
    };

    SUBCASE("groups = 1") {
      Conv2DAttrs attrs = Conv2DAttrs{
          /*out_channels=*/13_p,
          /*kernel_h=*/3_p,
          /*kernel_w=*/4_p,
          /*stride_h=*/1_p,
          /*stride_w=*/1_p,
          /*padding_h=*/0_n,
          /*padding_w=*/0_n,
          /*groups=*/1_p,
          /*activation=*/std::nullopt,
          /*use_bias=*/true,
      };

      TensorShape result = conv2d_get_output_shape(attrs, input);

      TensorShape correct = TensorShape{
          TensorDims{FFOrdered{
              7_p,
              13_p,
              9_p,
              12_p,
          }},
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }

    SUBCASE("groups > 1 does not affect the output shape") {
      Conv2DAttrs groups_eq_one = Conv2DAttrs{
          /*out_channels=*/13_p,
          /*kernel_h=*/3_p,
          /*kernel_w=*/4_p,
          /*stride_h=*/1_p,
          /*stride_w=*/1_p,
          /*padding_h=*/0_n,
          /*padding_w=*/0_n,
          /*groups=*/2_p,
          /*activation=*/std::nullopt,
          /*use_bias=*/true,
      };

      Conv2DAttrs groups_gt_one = [&]() {
        Conv2DAttrs result = groups_eq_one;
        result.groups = 2_p;
        return result;
      }();

      TensorShape groups_eq_one_result =
          conv2d_get_output_shape(groups_eq_one, input);
      TensorShape groups_gt_one_result =
          conv2d_get_output_shape(groups_gt_one, input);

      CHECK(groups_eq_one_result == groups_gt_one_result);
    }
  }

  TEST_CASE("conv2d_get_kernel_parallel_dim_degrees") {
    SUBCASE("groups = 1") {
      Conv2DAttrs attrs = Conv2DAttrs{
          /*out_channels=*/24_p,
          /*kernel_h=*/3_p,
          /*kernel_w=*/4_p,
          /*stride_h=*/1_p,
          /*stride_w=*/1_p,
          /*padding_h=*/0_n,
          /*padding_w=*/0_n,
          /*groups=*/1_p,
          /*activation=*/std::nullopt,
          /*use_bias=*/true,
      };

      SUBCASE("data parallelism") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{4_p, 1_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            conv2d_get_kernel_parallel_dim_degrees(attrs, input_degrees);

        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{4_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 1_p, 1_p, 1_p},
        };

        CHECK(result == correct);
      }

      SUBCASE("activation height parallelism") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 1_p, 2_p, 1_p},
        };

        // TODO: this should eventually be enabled, once we've figured out how
        // to handle the ghost points
        //
        // ParallelTensorDimDegrees result = conv2d_get_kernel_parallel_dim_degrees(attrs, input_degrees);
        //
        // ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
        //   /*sum_degree=*/SumDegree{1_p},
        //   /*discard_copy_degree=*/DiscardCopyDegree{2_p},
        //   /*shard_degrees=*/FFOrdered<positive_int>{1_p, 1_p, 1_p, 1_p},
        // };
        //
        // CHECK(result == correct);

        CHECK_THROWS(
            conv2d_get_kernel_parallel_dim_degrees(attrs, input_degrees));
      }

      SUBCASE("activation width parallelism") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 1_p, 1_p, 2_p},
        };

        // TODO: this should eventually be enabled, once we've figured out how
        // to handle the ghost points
        //
        // ParallelTensorDimDegrees result = conv2d_get_kernel_parallel_dim_degrees(attrs, input_degrees);
        //
        // ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
        //   /*sum_degree=*/SumDegree{1_p},
        //   /*discard_copy_degree=*/DiscardCopyDegree{2_p},
        //   /*shard_degrees=*/FFOrdered<positive_int>{1_p, 1_p, 1_p, 1_p},
        // };
        //
        // CHECK(result == correct);

        CHECK_THROWS(
            conv2d_get_kernel_parallel_dim_degrees(attrs, input_degrees));
      }

      SUBCASE("input channel parallelism") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 4_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            conv2d_get_kernel_parallel_dim_degrees(attrs, input_degrees);

        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 4_p, 1_p, 1_p},
        };

        CHECK(result == correct);
      }

      SUBCASE("output channel parallelism") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{4_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 1_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            conv2d_get_kernel_parallel_dim_degrees(attrs, input_degrees);

        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{4_p, 1_p, 1_p, 1_p},
        };

        CHECK(result == correct);
      }

      SUBCASE("propagating sum degree") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{4_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 1_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            conv2d_get_kernel_parallel_dim_degrees(attrs, input_degrees);

        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{4_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 1_p, 1_p, 1_p},
        };

        CHECK(result == correct);
      }
    }

    SUBCASE("groups > 1") {
      Conv2DAttrs attrs = Conv2DAttrs{
          /*out_channels=*/24_p,
          /*kernel_h=*/3_p,
          /*kernel_w=*/3_p,
          /*stride_h=*/1_p,
          /*stride_w=*/1_p,
          /*padding_h=*/0_n,
          /*padding_w=*/0_n,
          /*groups=*/4_p,
          /*activation=*/std::nullopt,
          /*use_bias=*/true,
      };

      SUBCASE("groups == input channel degree") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 4_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            conv2d_get_kernel_parallel_dim_degrees(attrs, input_degrees);

        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{4_p, 1_p, 1_p, 1_p},
        };

        CHECK(result == correct);
      }

      SUBCASE("groups is divisible by input channel degree") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 2_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            conv2d_get_kernel_parallel_dim_degrees(attrs, input_degrees);

        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{2_p, 1_p, 1_p, 1_p},
        };

        CHECK(result == correct);
      }

      SUBCASE("input channel degree is divisible by groups") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 24_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            conv2d_get_kernel_parallel_dim_degrees(attrs, input_degrees);

        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{4_p, 6_p, 1_p, 1_p},
        };

        CHECK(result == correct);
      }

      SUBCASE("input channel degree and groups are incompatible") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 6_p, 1_p, 1_p},
        };

        CHECK_THROWS(
            conv2d_get_kernel_parallel_dim_degrees(attrs, input_degrees));
      }
    }
  }

  TEST_CASE("conv2d_get_bias_parallel_dim_degrees") {
    SUBCASE("groups = 1") {
      Conv2DAttrs attrs = Conv2DAttrs{
          /*out_channels=*/24_p,
          /*kernel_h=*/3_p,
          /*kernel_w=*/4_p,
          /*stride_h=*/1_p,
          /*stride_w=*/1_p,
          /*padding_h=*/0_n,
          /*padding_w=*/0_n,
          /*groups=*/1_p,
          /*activation=*/std::nullopt,
          /*use_bias=*/true,
      };

      SUBCASE("data parallelism") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{4_p, 1_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            conv2d_get_bias_parallel_dim_degrees(attrs, input_degrees);

        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{4_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p},
        };

        CHECK(result == correct);
      }

      SUBCASE("activation height parallelism") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 1_p, 2_p, 1_p},
        };

        // TODO: this should eventually be enabled, once we've figured out how
        // to handle the ghost points
        //
        // ParallelTensorDimDegrees result = conv2d_get_bias_parallel_dim_degrees(attrs, input_degrees);
        //
        // ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
        //   /*sum_degree=*/SumDegree{2_p},
        //   /*discard_copy_degree=*/DiscardCopyDegree{1_p},
        //   /*shard_degrees=*/FFOrdered<positive_int>{1_p},
        // };
        //
        // CHECK(result == correct);

        CHECK_THROWS(
            conv2d_get_bias_parallel_dim_degrees(attrs, input_degrees));
      }

      SUBCASE("activation width parallelism") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 1_p, 1_p, 2_p},
        };

        // TODO: this should eventually be enabled, once we've figured out how
        // to handle the ghost points
        //
        // ParallelTensorDimDegrees result = conv2d_get_bias_parallel_dim_degrees(attrs, input_degrees);
        //
        // ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
        //   /*sum_degree=*/SumDegree{2_p},
        //   /*discard_copy_degree=*/DiscardCopyDegree{1_p},
        //   /*shard_degrees=*/FFOrdered<positive_int>{1_p},
        // };
        //
        // CHECK(result == correct);

        CHECK_THROWS(
            conv2d_get_bias_parallel_dim_degrees(attrs, input_degrees));
      }

      SUBCASE("input channel parallelism") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 4_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            conv2d_get_bias_parallel_dim_degrees(attrs, input_degrees);

        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{4_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p},
        };

        CHECK(result == correct);
      }

      SUBCASE("output channel parallelism") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{4_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 1_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            conv2d_get_bias_parallel_dim_degrees(attrs, input_degrees);

        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{4_p},
        };

        CHECK(result == correct);
      }

      SUBCASE("propagating sum degree") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{4_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 1_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            conv2d_get_bias_parallel_dim_degrees(attrs, input_degrees);

        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{4_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p},
        };

        CHECK(result == correct);
      }
    }

    SUBCASE("groups > 1") {
      Conv2DAttrs attrs = Conv2DAttrs{
          /*out_channels=*/24_p,
          /*kernel_h=*/3_p,
          /*kernel_w=*/3_p,
          /*stride_h=*/1_p,
          /*stride_w=*/1_p,
          /*padding_h=*/0_n,
          /*padding_w=*/0_n,
          /*groups=*/4_p,
          /*activation=*/std::nullopt,
          /*use_bias=*/true,
      };

      SUBCASE("groups == input channel degree") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 4_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            conv2d_get_bias_parallel_dim_degrees(attrs, input_degrees);

        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{4_p},
        };

        CHECK(result == correct);
      }

      SUBCASE("groups is divisible by input channel degree") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 2_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            conv2d_get_bias_parallel_dim_degrees(attrs, input_degrees);

        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{2_p},
        };

        CHECK(result == correct);
      }

      SUBCASE("input channel degree is divisible by groups") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 24_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            conv2d_get_bias_parallel_dim_degrees(attrs, input_degrees);

        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{6_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{4_p},
        };

        CHECK(result == correct);
      }

      SUBCASE("input channel degree and groups are incompatible") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 6_p, 1_p, 1_p},
        };

        CHECK_THROWS(
            conv2d_get_bias_parallel_dim_degrees(attrs, input_degrees));
      }
    }
  }

  TEST_CASE("conv2d_get_output_parallel_dim_degrees") {
    SUBCASE("groups = 1") {
      Conv2DAttrs attrs = Conv2DAttrs{
          /*out_channels=*/24_p,
          /*kernel_h=*/3_p,
          /*kernel_w=*/3_p,
          /*stride_h=*/1_p,
          /*stride_w=*/1_p,
          /*padding_h=*/0_n,
          /*padding_w=*/0_n,
          /*groups=*/1_p,
          /*activation=*/std::nullopt,
          /*use_bias=*/true,
      };

      SUBCASE("data parallelism") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{4_p, 1_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            conv2d_get_output_parallel_dim_degrees(attrs, input_degrees);

        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{4_p, 1_p, 1_p, 1_p},
        };

        CHECK(result == correct);
      }

      SUBCASE("activation height parallelism") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 1_p, 2_p, 1_p},
        };

        // TODO: this should eventually be enabled, once we've figured out how
        // to handle the ghost points
        //
        // ParallelTensorDimDegrees result = conv2d_get_output_parallel_dim_degrees(attrs, input_degrees);
        //
        // ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
        //   /*sum_degree=*/SumDegree{1_p},
        //   /*discard_copy_degree=*/DiscardCopyDegree{1_p},
        //   /*shard_degrees=*/FFOrdered<positive_int>{1_p, 1_p, 2_p, 1_p},
        // };
        //
        // CHECK(result == correct);

        CHECK_THROWS(
            conv2d_get_output_parallel_dim_degrees(attrs, input_degrees));
      }

      SUBCASE("activation width parallelism") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 1_p, 1_p, 2_p},
        };

        // TODO: this should eventually be enabled, once we've figured out how
        // to handle the ghost points
        //
        // ParallelTensorDimDegrees result = conv2d_get_output_parallel_dim_degrees(attrs, input_degrees);
        //
        // ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
        //   /*sum_degree=*/SumDegree{1_p},
        //   /*discard_copy_degree=*/DiscardCopyDegree{1_p},
        //   /*shard_degrees=*/FFOrdered<positive_int>{1_p, 1_p, 1_p, 2_p},
        // };
        //
        // CHECK(result == correct);

        CHECK_THROWS(
            conv2d_get_output_parallel_dim_degrees(attrs, input_degrees));
      }

      SUBCASE("input channel parallelism") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 4_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            conv2d_get_output_parallel_dim_degrees(attrs, input_degrees);

        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{4_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 1_p, 1_p, 1_p},
        };

        CHECK(result == correct);
      }

      SUBCASE("output channel parallelism") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{4_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 1_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            conv2d_get_output_parallel_dim_degrees(attrs, input_degrees);

        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 4_p, 1_p, 1_p},
        };

        CHECK(result == correct);
      }

      SUBCASE("propagating sum degree") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{4_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 1_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            conv2d_get_output_parallel_dim_degrees(attrs, input_degrees);

        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{4_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 1_p, 1_p, 1_p},
        };

        CHECK(result == correct);
      }
    }

    SUBCASE("groups > 1") {
      Conv2DAttrs attrs = Conv2DAttrs{
          /*out_channels=*/24_p,
          /*kernel_h=*/3_p,
          /*kernel_w=*/3_p,
          /*stride_h=*/1_p,
          /*stride_w=*/1_p,
          /*padding_h=*/0_n,
          /*padding_w=*/0_n,
          /*groups=*/4_p,
          /*activation=*/std::nullopt,
          /*use_bias=*/true,
      };

      SUBCASE("groups == input channel degree") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 4_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            conv2d_get_output_parallel_dim_degrees(attrs, input_degrees);

        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 4_p, 1_p, 1_p},
        };

        CHECK(result == correct);
      }

      SUBCASE("groups is divisible by input channel degree") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 2_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            conv2d_get_output_parallel_dim_degrees(attrs, input_degrees);

        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 2_p, 1_p, 1_p},
        };

        CHECK(result == correct);
      }

      SUBCASE("input channel degree is divisible by groups") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 24_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            conv2d_get_output_parallel_dim_degrees(attrs, input_degrees);

        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{6_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 4_p, 1_p, 1_p},
        };

        CHECK(result == correct);
      }

      SUBCASE("input channel degree and groups are incompatible") {
        ParallelTensorDimDegrees input_degrees = ParallelTensorDimDegrees{
            /*sum_degree=*/SumDegree{1_p},
            /*discard_copy_degree=*/DiscardCopyDegree{1_p},
            /*shard_degrees=*/FFOrdered<positive_int>{1_p, 6_p, 1_p, 1_p},
        };

        CHECK_THROWS(
            conv2d_get_output_parallel_dim_degrees(attrs, input_degrees));
      }
    }
  }

  TEST_CASE("Conv2D shape inference") {
    positive_int out_channels = 4_p;
    positive_int kernel_h = 3_p;
    positive_int kernel_w = 2_p;
    positive_int stride_h = 2_p;
    positive_int stride_w = 2_p;
    nonnegative_int padding_h = 1_n;
    nonnegative_int padding_w = 1_n;
    positive_int groups = 1_p;
    std::optional<Activation> activation = std::nullopt;
    bool use_bias = true;

    Conv2DAttrs attrs = Conv2DAttrs{
        /*out_channels=*/out_channels,
        /*kernel_h=*/kernel_h,
        /*kernel_w=*/kernel_w,
        /*stride_h=*/stride_h,
        /*stride_w=*/stride_w,
        /*padding_h=*/padding_h,
        /*padding_w=*/padding_w,
        /*groups=*/groups,
        /*activation=*/activation,
        /*use_bias=*/true,
    };

    positive_int num_samples = 7_p;
    positive_int input_channels = 4_p;
    positive_int input_height = 11_p;
    positive_int input_width = 15_p;

    TensorShape input = TensorShape{
        TensorDims{FFOrdered{
            num_samples,
            input_channels,
            input_height,
            input_width,
        }},
        DataType::FLOAT,
    };

    positive_int output_height = 6_p;
    positive_int output_width = 8_p;

    TensorShape output = TensorShape{
        TensorDims{FFOrdered{
            num_samples,
            out_channels,
            output_height,
            output_width,
        }},
        DataType::FLOAT,
    };

    TensorShape kernel = TensorShape{
        TensorDims{FFOrdered{
            out_channels,
            input_channels,
            kernel_h,
            kernel_w,
        }},
        DataType::FLOAT,
    };

    TensorShape bias = TensorShape{
        TensorDims{FFOrdered{
            out_channels,
        }},
        DataType::FLOAT,
    };

    SUBCASE("conv2d_get_output_shape") {
      TensorShape result_output = conv2d_get_output_shape(attrs, input);
      TensorShape correct_output = output;
      CHECK(result_output == correct_output);
    }

    SUBCASE("conv2d_get_kernel_shape") {
      TensorShape result_kernel = conv2d_get_kernel_shape(attrs, input);
      TensorShape correct_kernel = kernel;
      CHECK(result_kernel == correct_kernel);
    }

    SUBCASE("conv2d_get_bias_shape") {
      TensorShape result_bias = conv2d_get_bias_shape(attrs, input);
      TensorShape correct_bias = bias;
      CHECK(result_bias == correct_bias);
    }

    auto make_input = [&](SumDegree o_sum,
                          DiscardCopyDegree o_eq,
                          positive_int o_n,
                          positive_int o_c,
                          positive_int o_h,
                          positive_int o_w) {
      return lift_to_parallel_with_degrees(
          input, o_sum, o_eq, FFOrdered{o_n, o_c, o_h, o_w});
    };

    auto make_output = [&](SumDegree o_sum,
                           DiscardCopyDegree o_eq,
                           positive_int o_n,
                           positive_int o_c,
                           positive_int o_h,
                           positive_int o_w) {
      return lift_to_parallel_with_degrees(
          output, o_sum, o_eq, FFOrdered{o_n, o_c, o_h, o_w});
    };

    auto make_kernel = [&](SumDegree o_sum,
                           DiscardCopyDegree o_eq,
                           positive_int o_outchannels,
                           positive_int o_inchannels,
                           positive_int o_kernel_h,
                           positive_int o_kernel_w) {
      return lift_to_parallel_with_degrees(
          kernel,
          o_sum,
          o_eq,
          FFOrdered{o_outchannels, o_inchannels, o_kernel_h, o_kernel_w});
    };

    auto make_bias = [&](SumDegree o_sum,
                         DiscardCopyDegree o_eq,
                         positive_int o_outchannels) {
      return lift_to_parallel_with_degrees(
          bias, o_sum, o_eq, FFOrdered{o_outchannels});
    };

    SUBCASE("data parallelism") {
      positive_int degree = 2_p;
      ParallelTensorShape par_input = make_input(
          SumDegree{1_p}, DiscardCopyDegree{1_p}, degree, 1_p, 1_p, 1_p);

      SUBCASE("get_output_parallel_shape") {
        ParallelTensorShape result =
            conv2d_get_output_parallel_shape(attrs, par_input);
        ParallelTensorShape correct = make_output(
            SumDegree{1_p}, DiscardCopyDegree{1_p}, degree, 1_p, 1_p, 1_p);
        CHECK(result == correct);
      }

      SUBCASE("get_kernel_parallel_shape") {
        ParallelTensorShape result =
            conv2d_get_kernel_parallel_shape(attrs, par_input);
        ParallelTensorShape correct = make_kernel(
            SumDegree{1_p}, DiscardCopyDegree{degree}, 1_p, 1_p, 1_p, 1_p);
        CHECK(result == correct);
      }

      SUBCASE("get_bias_parallel_shape") {
        ParallelTensorShape result =
            conv2d_get_bias_parallel_shape(attrs, par_input);
        ParallelTensorShape correct =
            make_bias(SumDegree{1_p}, DiscardCopyDegree{degree}, 1_p);
        CHECK(result == correct);
      }
    }

    SUBCASE("input channel parallelism") {
      positive_int degree = 2_p;
      ParallelTensorShape par_input = make_input(
          SumDegree{1_p}, DiscardCopyDegree{1_p}, 1_p, degree, 1_p, 1_p);

      SUBCASE("conv2d_get_output_parallel_shape") {
        ParallelTensorShape result =
            conv2d_get_output_parallel_shape(attrs, par_input);
        ParallelTensorShape correct = make_output(
            SumDegree{degree}, DiscardCopyDegree{1_p}, 1_p, 1_p, 1_p, 1_p);
        CHECK(result == correct);
      }

      SUBCASE("conv2d_get_kernel_parallel_shape") {
        ParallelTensorShape result =
            conv2d_get_kernel_parallel_shape(attrs, par_input);
        ParallelTensorShape correct = make_kernel(
            SumDegree{1_p}, DiscardCopyDegree{1_p}, 1_p, degree, 1_p, 1_p);
        CHECK(result == correct);
      }

      SUBCASE("conv2d_get_bias_parallel_shape") {
        ParallelTensorShape result =
            conv2d_get_bias_parallel_shape(attrs, par_input);
        ParallelTensorShape correct =
            make_bias(SumDegree{degree}, DiscardCopyDegree{1_p}, 1_p);
        CHECK(result == correct);
      }
    }

    SUBCASE("output channel parallelism") {
      positive_int degree = 2_p;
      ParallelTensorShape par_input = make_input(
          SumDegree{1_p}, DiscardCopyDegree{degree}, 1_p, 1_p, 1_p, 1_p);

      SUBCASE("conv2d_get_output_parallel_shape") {
        ParallelTensorShape result =
            conv2d_get_output_parallel_shape(attrs, par_input);
        ParallelTensorShape correct = make_output(
            SumDegree{1_p}, DiscardCopyDegree{1_p}, 1_p, degree, 1_p, 1_p);
        CHECK(result == correct);
      }

      SUBCASE("conv2d_get_kernel_parallel_shape") {
        ParallelTensorShape result =
            conv2d_get_kernel_parallel_shape(attrs, par_input);
        ParallelTensorShape correct = make_kernel(
            SumDegree{1_p}, DiscardCopyDegree{1_p}, degree, 1_p, 1_p, 1_p);
        CHECK(result == correct);
      }

      SUBCASE("conv2d_get_bias_parallel_shape") {
        ParallelTensorShape result =
            conv2d_get_bias_parallel_shape(attrs, par_input);
        ParallelTensorShape correct =
            make_bias(SumDegree{1_p}, DiscardCopyDegree{1_p}, degree);
        CHECK(result == correct);
      }
    }

    SUBCASE("propagating sum degree") {
      positive_int degree = 2_p;
      ParallelTensorShape par_input = make_input(
          SumDegree{degree}, DiscardCopyDegree{1_p}, 1_p, 1_p, 1_p, 1_p);

      SUBCASE("conv2d_get_output_parallel_shape") {
        ParallelTensorShape result =
            conv2d_get_output_parallel_shape(attrs, par_input);
        ParallelTensorShape correct = make_output(
            SumDegree{degree}, DiscardCopyDegree{1_p}, 1_p, 1_p, 1_p, 1_p);
        CHECK(result == correct);
      }

      SUBCASE("conv2d_get_kernel_parallel_shape") {
        ParallelTensorShape result =
            conv2d_get_kernel_parallel_shape(attrs, par_input);
        ParallelTensorShape correct = make_kernel(
            SumDegree{1_p}, DiscardCopyDegree{degree}, 1_p, 1_p, 1_p, 1_p);
        CHECK(result == correct);
      }

      SUBCASE("conv2d_get_bias_parallel_shape") {
        ParallelTensorShape result =
            conv2d_get_bias_parallel_shape(attrs, par_input);
        ParallelTensorShape correct =
            make_bias(SumDegree{degree}, DiscardCopyDegree{1_p}, 1_p);
        CHECK(result == correct);
      }
    }
  }

  TEST_CASE("conv2d_get_shard_signature_instance") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    Conv2DAttrs attrs = Conv2DAttrs{
      /*out_channels=*/12_p,
      /*kernel_h=*/3_p,
      /*kernel_w=*/3_p,
      /*stride_h=*/1_p,
      /*stride_w=*/1_p,
      /*padding_h=*/0_n,
      /*padding_w=*/0_n,
      /*groups=*/1_p,
      /*activation=*/std::nullopt,
      /*use_bias=*/false,
    };

    TensorShape input_shape = TensorShape{
      TensorDims{
        FFOrdered{
          6_p,
          4_p,
          5_p,
          3_p,
        },
      },
      DataType::FLOAT,
    };

    auto mk_dim_degrees = [&](int sum_degree,
                              int discard_copy_degree,
                              int batch_shard_degree,
                              int channel_shard_degree,
                              int height_shard_degree,
                              int width_shard_degree)
      -> ParallelTensorDimDegrees
    {
      return ParallelTensorDimDegrees{
        /*sum_degree=*/SumDegree{positive_int{sum_degree}},
        /*discard_copy_degree=*/DiscardCopyDegree{positive_int{discard_copy_degree}},
        /*shard_degrees=*/FFOrdered{
          positive_int{batch_shard_degree},
          positive_int{channel_shard_degree},
          positive_int{height_shard_degree},
          positive_int{width_shard_degree},
        },
      };
    };

    auto run_conv2d = [&](std::map<TensorSlotName, GenericTensorAccessorR> const &incoming_shards)
      -> std::map<TensorSlotName, GenericTensorAccessorR>
    {
      GenericTensorAccessorR input_shard = incoming_shards.at(TensorSlotName::INPUT);
      TensorShape output_shard_shape =
        conv2d_get_output_shape(attrs, get_tensor_shape_for_accessor_r(input_shard));
      GenericTensorAccessorW output_shard = create_zero_filled_accessor_w(output_shard_shape, cpu_allocator);

      conv2d_cpu_forward_kernel(
        /*attrs=*/attrs,
        /*input=*/input_shard,
        /*output=*/output_shard);

      return std::map<TensorSlotName, GenericTensorAccessorR>{
        {
          TensorSlotName::OUTPUT,
          read_only_accessor_from_write_accessor(output_shard),
        },
      };
    };

    auto conv2d_shard_signature_instance_is_valid = [&](ParallelTensorDimDegrees const &input_degrees)
      -> bool
    {
      ParallelTensorShape input_parallel_shape = lift_to_parallel_with_degrees(input_shape, input_degrees);

      std::map<TensorSlotName, ParallelTensorShape> input_shapes = {
        {
          TensorSlotName::INPUT,
          input_parallel_shape,
        },
      };

      return shard_signature_instance_is_valid(
        /*attrs=*/ComputationGraphOpAttrs{attrs},
        /*input_shapes=*/input_shapes,
        /*run_op=*/run_conv2d,
        /*seed=*/0);
    };

    SUBCASE("data parallelism") {
      ParallelTensorDimDegrees input_dim_degrees = mk_dim_degrees(1, 1, 2, 1, 1, 1);

      CHECK(conv2d_shard_signature_instance_is_valid(input_dim_degrees));
    }

    SUBCASE("mixed parallelism") {
      ParallelTensorDimDegrees input_dim_degrees = mk_dim_degrees(3, 2, 3, 2, 1, 1);

      CHECK(conv2d_shard_signature_instance_is_valid(input_dim_degrees));
    }
  }
}
