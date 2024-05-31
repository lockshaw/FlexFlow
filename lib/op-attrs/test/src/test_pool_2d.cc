#include "doctest/doctest.h"
#include "op-attrs/ops/pool_2d.h"

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_output_shape(Pool2DAttrs, TensorShape)") {

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

    TensorShape result = get_output_shape(attrs, input_shape);

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

    CHECK(result == correct_output_shape);
  }
}
