#include "op-attrs/ops/reshape.h"
#include <doctest/doctest.h>
#include "kernels/reshape_kernels_cpu.h"
#include "kernels/accessor.h"
#include "kernels/create_zero_filled_accessor.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "kernels/shard_signature_instance_is_valid.h"
#include "kernels/local_cpu_allocator.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("reshape_get_output_shape") {
    ReshapeAttrs attrs = ReshapeAttrs{
      TensorDims{FFOrdered<positive_int>{6_p, 5_p}},
      TensorDims{FFOrdered<positive_int>{3_p, 10_p}},
    };

    SUBCASE("input tensor has different num elements") {
      TensorShape input = TensorShape{
          TensorDims{FFOrdered<positive_int>{4_p, 5_p, 5_p}},
          DataType::FLOAT,
      };

      CHECK_THROWS(reshape_get_output_shape(attrs, input));
    }

    SUBCASE("input tensor has different datatype") {
      TensorShape input = TensorShape{
          TensorDims{FFOrdered<positive_int>{4_p, 6_p, 5_p}},
          DataType::DOUBLE,
      };

      TensorShape result = reshape_get_output_shape(attrs, input);
      TensorShape correct = TensorShape{
          TensorDims{FFOrdered<positive_int>{4_p, 3_p, 10_p}},
          DataType::DOUBLE,
      };

      CHECK(result == correct);
    }

    SUBCASE("valid input") {
      TensorShape input = TensorShape{
          TensorDims{FFOrdered<positive_int>{4_p, 6_p, 5_p}},
          DataType::FLOAT,
      };

      TensorShape result = reshape_get_output_shape(attrs, input);

      TensorShape correct = TensorShape{
          TensorDims{FFOrdered<positive_int>{4_p, 3_p, 10_p}},
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }

    SUBCASE("input tensor exactly matches core input dims") {
      TensorShape input = TensorShape{
          TensorDims{FFOrdered<positive_int>{6_p, 5_p}},
          DataType::FLOAT,
      };

      TensorShape result = reshape_get_output_shape(attrs, input);

      TensorShape correct = TensorShape{
          TensorDims{FFOrdered<positive_int>{3_p, 10_p}},
          DataType::FLOAT,
      };

      CHECK(result == correct);
    }

    SUBCASE("input tensor has fewer dims than core input dims") {
      TensorShape input = TensorShape{
          TensorDims{FFOrdered<positive_int>{5_p}},
          DataType::FLOAT,
      };

      CHECK_THROWS(reshape_get_output_shape(attrs, input));
    }
  }

  TEST_CASE(
      "reshape_get_output_parallel_dim_degrees") {
    
    SUBCASE("preserves number of dims") {
      ReshapeAttrs attrs = ReshapeAttrs{
        TensorDims{FFOrdered<positive_int>{12_p, 5_p}},
        TensorDims{FFOrdered<positive_int>{6_p, 10_p}},
      };

      SUBCASE("input sum degree > 1") {
        ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
            SumDegree{3_p},
            DiscardCopyDegree{1_p},
            FFOrdered<positive_int>{1_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            reshape_get_output_parallel_dim_degrees(attrs, input);
        ParallelTensorDimDegrees correct = input;

        CHECK(result == correct);
      }

      SUBCASE("input discard copy degree > 1") {
        ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
            SumDegree{1_p},
            DiscardCopyDegree{2_p},
            FFOrdered<positive_int>{1_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            reshape_get_output_parallel_dim_degrees(attrs, input);
        ParallelTensorDimDegrees correct = input;

        CHECK(result == correct);
      }

      SUBCASE("allows leading shard degree") {
        ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
            SumDegree{1_p},
            DiscardCopyDegree{1_p},
            FFOrdered<positive_int>{2_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            reshape_get_output_parallel_dim_degrees(attrs, input);
        ParallelTensorDimDegrees correct = input;

        CHECK(result == correct);
      }

      SUBCASE("does not allow shard degree in core") {
        ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
            SumDegree{1_p},
            DiscardCopyDegree{1_p},
            FFOrdered<positive_int>{1_p, 6_p, 1_p},
        };

        CHECK_THROWS(reshape_get_output_parallel_dim_degrees(attrs, input));
      }
    }

    SUBCASE("changes number of dims") {
      ReshapeAttrs attrs = ReshapeAttrs{
        TensorDims{FFOrdered<positive_int>{6_p, 5_p}},
        TensorDims{FFOrdered<positive_int>{30_p}},
      };

      SUBCASE("input sum degree > 1") {
        ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
            SumDegree{3_p},
            DiscardCopyDegree{1_p},
            FFOrdered<positive_int>{1_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            reshape_get_output_parallel_dim_degrees(attrs, input);
        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            SumDegree{3_p},
            DiscardCopyDegree{1_p},
            FFOrdered<positive_int>{1_p, 1_p},
        };

        CHECK(result == correct);
      }

      SUBCASE("input discard copy degree > 1") {
        ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
            SumDegree{1_p},
            DiscardCopyDegree{2_p},
            FFOrdered<positive_int>{1_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            reshape_get_output_parallel_dim_degrees(attrs, input);
        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            SumDegree{1_p},
            DiscardCopyDegree{2_p},
            FFOrdered<positive_int>{1_p, 1_p},
        };

        CHECK(result == correct);
      }

      SUBCASE("allows leading shard degree") {
        ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
            SumDegree{1_p},
            DiscardCopyDegree{1_p},
            FFOrdered<positive_int>{2_p, 1_p, 1_p},
        };

        ParallelTensorDimDegrees result =
            reshape_get_output_parallel_dim_degrees(attrs, input);
        ParallelTensorDimDegrees correct = ParallelTensorDimDegrees{
            SumDegree{1_p},
            DiscardCopyDegree{1_p},
            FFOrdered<positive_int>{2_p, 1_p},
        };

        CHECK(result == correct);
      }

      SUBCASE("does not allow shard degree in core") {
        ParallelTensorDimDegrees input = ParallelTensorDimDegrees{
            SumDegree{1_p},
            DiscardCopyDegree{1_p},
            FFOrdered<positive_int>{1_p, 6_p, 1_p},
        };

        CHECK_THROWS(reshape_get_output_parallel_dim_degrees(attrs, input));
      }
    }
  }

  TEST_CASE("reshape_get_shard_signature_instance") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    ReshapeAttrs attrs = ReshapeAttrs{
      /*core_input_dims=*/TensorDims{
        FFOrdered{
          4_p, 5_p,
        },
      },
      /*core_output_dims=*/TensorDims{
        FFOrdered{
          2_p, 10_p,
        },
      },
    };

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

    auto run_reshape = [&](std::map<TensorSlotName, GenericTensorAccessorR> const &incoming_shards)
      -> std::map<TensorSlotName, GenericTensorAccessorR>
    {
      GenericTensorAccessorR input_shard = incoming_shards.at(TensorSlotName::INPUT);
      TensorShape output_shard_shape =
        reshape_get_output_shape(attrs, get_tensor_shape_for_accessor_r(input_shard));
      GenericTensorAccessorW output_shard = create_zero_filled_accessor_w(output_shard_shape, cpu_allocator);

      reshape_cpu_forward_kernel(
        /*input=*/input_shard,
        /*output=*/output_shard);

      return std::map<TensorSlotName, GenericTensorAccessorR>{
        {
          TensorSlotName::OUTPUT,
          read_only_accessor_from_write_accessor(output_shard),
        },
      };
    };

    auto reshape_shard_signature_instance_is_valid = [&](ParallelTensorDimDegrees const &input_degrees)
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
        /*run_op=*/run_reshape,
        /*seed=*/0);
    };

    SUBCASE("data parallelism") {
      ParallelTensorDimDegrees input_dim_degrees = mk_dim_degrees(1, 1, 2, 1, 1);

      CHECK(reshape_shard_signature_instance_is_valid(input_dim_degrees));
    }
  }
}
