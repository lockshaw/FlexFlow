#include <doctest/doctest.h>
#include "kernels/conv_2d_kernels_cpu.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("conv2d_cpu_forward_kernel") {
    GenericTensorAccessorR input = create_4d_accessor_r_with_contents<float>(
      {
        {
          {
            {3, 9, 3, -3, 1},
            {5, 4, 3, 9, 1},
            {4, 2, 6, 0, 5},
          },
          {
            {-1, 1, -1, 9, -2},
            {6, 9, 1, 5, 8},
            {9, 6, -1, 1, -2},
          },
        },
        {
          {
            {8, -2, 7, 2, 4},
            {5, -2, 2, 3, 2},
            {6, 7, 0, 5, 4},
          },
          {
            {4, 5, 1, -3, 9},
            {5, -3, -2, 8, 3},
            {8, 9, 7, 7, -3},
          },
        },
      },
      cpu_allocator);

    GenericTensorAccessorR filter = create_4d_accessor_r_with_contents<float>(
      {
        {
          {
            {6, 4, 2},
            {0, 8, 2},
          },
          {
            {8, -2, 0},
            {6, 0, 0},
          },
        },
        {
          {
            {9, -1, 9},
            {5, 4, -2},
          },
          {
            {-2, 2, 5},
            {4, -2, 1},
          },
        },
        {
          {
            {5, 1, 8},
            {-2, 5, 2},
          },
          {
            {5, 0, 9},
            {6, 5, 6},
          },
        },
      },
      cpu_allocator);

    GenericTensorAccessorR bias = create_1d_accessor_r_with_contents<float>(
      {1, 4, -2},
      cpu_allocator);

    GenericTensorAccessorR correct = create_4d_accessor_r_with_contents<float>(
      {
        {
          {
            {125, 167,  63},
            {165, 209,  59},
          },
          {
            { 90, 149, 104},
            {122, 188,  91},
          },
          {
            {135, 222, 115},
            {182, 246,  92},
          },
        },
        {
          {
            { 95,  63,  89},
            {177,  47,  87},
          },
          {
            {185, -34, 135},
            {122, 107,  95},
          },
          {
            {108,  49, 214},
            {202, 198, 136},
          },
        },
      },
      cpu_allocator);

    GenericTensorAccessorW result = create_zero_filled_accessor_w(correct.shape, cpu_allocator);

    Conv2DAttrs attrs = Conv2DAttrs{
      /*out_channels=*/2_p,
      /*kernel_h=*/2_p,
      /*kernel_w=*/3_p,
      /*stride_h=*/1_p,
      /*stride_w=*/1_p,
      /*padding_h=*/0_n,
      /*padding_w=*/0_n,
      /*groups=*/1_p,
      /*activation=*/std::nullopt,
      /*use_bias=*/true,
    };

    conv2d_cpu_forward_kernel(
      /*attrs=*/attrs,
      /*input=*/input,
      /*filter=*/filter,
      /*bias=*/bias,
      /*output=*/result);

    CHECK_MESSAGE(accessors_are_equal(result, correct),
                  check_kv("result", format_accessor_w_contents(result)));
  }

  TEST_CASE("conv2d_cpu_backward_kernel") {
    // TODO(@lockshaw)(#pr):
    NOT_IMPLEMENTED();
  }
}
