#include "doctest/doctest.h"
#include "op-attrs/ops/pool_2d.h"
#include "op-attrs/parallel_tensor_shape.h"

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("Pool 2D parallel shape inference - across dimensions") {

    int kernel_h = 6;
    int kernel_w = 4;
    int stride_h = 1;
    int stride_w = 3;
    int padding_h = 1;
    int padding_w = 1;
    std::optional<Activation> activation = std::nullopt;
    PoolOp pool_type = PoolOp::MAX;
    Pool2DAttrs attrs = {
      /*kernel_h=*/kernel_h,
      /*kernel_w=*/kernel_w,
      /*stride_h=*/stride_h,
      /*stride_w=*/stride_w,
      /*padding_h=*/padding_h,
      /*padding_w=*/padding_w,
     /**pool_type=*/pool_type,
      /*activation=*/activation
    };

    size_t num_samples = 10;
    size_t input_channels = 6;
    size_t input_height = 10;
    size_t input_width = 8;
    TensorShape input_shape = {
      TensorDims{
        FFOrdered<size_t>{
          num_samples,
          input_channels,
          input_height,
          input_width,
        }
      },
      DataType::FLOAT,
    };

    size_t correct_output_height = 7;
    size_t correct_output_width = 3;
    TensorShape correct_output_shape = {
      TensorDims{
        FFOrdered<size_t>{
          num_samples,     
          static_cast<size_t>(input_channels), 
          correct_output_height,
          correct_output_width,
        }
      },
      DataType::FLOAT,
    };

    size_t correct_kernel_height = kernel_h;
    size_t correct_kernel_width = kernel_w;
    TensorShape correct_kernel_shape = {
      TensorDims{
        FFOrdered<size_t>{
          correct_kernel_height,
          correct_kernel_width
        }
      },
      DataType::FLOAT,
    };

    TensorShape output_result = get_output_shape(attrs, input_shape);
    CHECK(output_result == correct_output_shape);

    TensorShape kernel_result = get_kernel_shape(attrs, input_shape);
    CHECK(kernel_result == correct_kernel_shape);




    auto make_input = [&](SumDegree o_sum, DiscardCopyDegree o_eq, int o_samples, int o_channels, int o_height, int o_width) {
      return lift_to_parallel_with_degrees(input_shape, o_sum, o_eq, FFOrdered<int>{o_samples, o_channels, o_height, o_width});
    };

    auto make_output = [&](SumDegree o_sum, DiscardCopyDegree o_eq, int o_samples, int o_channels, int o_height, int o_width) {
      return lift_to_parallel_with_degrees(correct_output_shape, o_sum, o_eq, FFOrdered<int>{o_samples, o_channels, o_height, o_width});
    };

    auto make_kernel = [&](SumDegree o_sum, DiscardCopyDegree o_eq, int o_height, int o_width) {
      return lift_to_parallel_with_degrees(correct_kernel_shape, o_sum, o_eq, FFOrdered<int>{o_height, o_width});
    };


    SUBCASE("data parallelism") {
      int degree = 4;
      ParallelTensorShape par_input = make_input(SumDegree{1}, DiscardCopyDegree{1}, degree, 1, input_height, input_width);

      {
        ParallelTensorShape result = get_output_shape(attrs, par_input);
        ParallelTensorShape correct = make_output(SumDegree{1}, DiscardCopyDegree{1}, degree, 1, correct_output_height, correct_output_width);
        CHECK(result == correct);
      }

      {
        ParallelTensorShape result = get_kernel_shape(attrs, par_input);
        ParallelTensorShape correct = make_kernel(SumDegree{1}, DiscardCopyDegree{degree}, correct_kernel_height, correct_kernel_width);
        CHECK(result == correct);
      }
    }

    SUBCASE("feature parallelism") {
      int degree = 4;
      ParallelTensorShape input = make_input(SumDegree{1}, DiscardCopyDegree{1}, 1, degree, input_height, input_width);

      {
        ParallelTensorShape result = get_output_shape(attrs, input);
        ParallelTensorShape correct = make_output(SumDegree{degree}, DiscardCopyDegree{1}, 1, 1, correct_output_height, correct_output_width);
        CHECK(result == correct);
      }

      {
        ParallelTensorShape result = get_kernel_shape(attrs, input);
        ParallelTensorShape correct = make_kernel(SumDegree{1}, DiscardCopyDegree{degree}, 1, 1, correct_kernel_height, correct_kernel_width);
        CHECK(result == correct);
      }
    }

    SUBCASE("output channel shard parallelism") {
      int degree = 4;
      ParallelTensorShape input = make_input(SumDegree{1}, DiscardCopyDegree{degree}, 1, 1, input_height, input_width);

      {
        ParallelTensorShape result = get_output_shape(attrs, input);
        ParallelTensorShape correct = make_output(SumDegree{1}, DiscardCopyDegree{1}, 1, degree, correct_output_height, correct_output_width);
        CHECK(result == correct);
      }

      {
        ParallelTensorShape result = get_kernel_shape(attrs, input);
        ParallelTensorShape correct = make_kernel(SumDegree{}, DiscardCopyDegree{degree}, 1, 1, correct_kernel_height, correct_kernel_width);
        CHECK(result == correct);
      }
    }
  }



  }

  TEST_CASE("Pool2D parallel shape inference") {

    int kernel_h = 6;
    int kernel_w = 4;
    int stride_h = 1;
    int stride_w = 3;
    int padding_h = 1;
    int padding_w = 1;
    std::optional<Activation> activation = std::nullopt;
    PoolOp pool_type = PoolOp::MAX;
    Pool2DAttrs attrs = {
      /*kernel_h=*/kernel_h,
      /*kernel_w=*/kernel_w,
      /*stride_h=*/stride_h,
      /*stride_w=*/stride_w,
      /*padding_h=*/padding_h,
      /*padding_w=*/padding_w,
     /**pool_type=*/pool_type,
      /*activation=*/activation
    };

    ParallelTensorShape input_shape = ParallelTensorShape{
      ParallelTensorDims{
        FFOrdered<ShardParallelDim>{
          ShardParallelDim{10, 2}, //sample
          ShardParallelDim{15, 3}, //channels
          ShardParallelDim{8, 1},  //height
          ShardParallelDim{9, 3}   //width
        },
        ReplicaParallelDimSet{
          SumDegree{3},
          DiscardCopyDegree{2}
        }
      },
      DataType::FLOAT,
    };

    size_t correct_output_height = 5;
    size_t correct_output_width = 3;
    size_t correct_sum_degree = 9;
    size_t correct_discard_copy_degree = 1;
    ParallelTensorShape correct_output_shape = ParallelTensorShape{
      ParallelTensorDims{
        FFOrdered<ShardParallelDim>{
          ShardParallelDim{10, 2}, //sample 
          ShardParallelDim{15, 2}, //channels
          ShardParallelDim{correct_output_height, 1},  //height
          ShardParallelDim{correct_output_width, 3}   //width
        },
        ReplicaParallelDimSet{
          SumDegree{correct_sum_degree},
          DiscardCopyDegree{correct_discard_copy_degree}
        }
      },
      DataType::FLOAT,
    };

    ParallelTensorShape output_result = get_output_shape(attrs, input_shape);
    CHECK(output_result == correct_output_shape);


  }

}
