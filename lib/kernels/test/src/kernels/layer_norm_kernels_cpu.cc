#include <doctest/doctest.h>
#include "kernels/layer_norm_kernels_cpu.h"
#include "kernels/local_cpu_allocator.h"
#include "kernels/create_zero_filled_accessor.h"
#include "kernels/accessors_are_equal.h"
#include "kernels/format_accessor_contents.h"
#include "test/utils/doctest/check_kv.h"
#include "kernels/create_accessor_with_contents.h"
#include "kernels/accessors_are_within_epsilon.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("layer_norm_cpu_forward_kernel") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR input =
        create_3d_accessor_r_with_contents<float>(
            {
                {
                    {3, 3, 6},
                    {2, 1, 5},
                    {1, 2, -2},
                    {8, 0.5, -3},
                },
                {
                    {5, 1, -2},
                    {-8, 0, -1},
                    {0.25, -0.3, -2},
                    {0, -1, -5},
                },
            },
            cpu_allocator);

    GenericTensorAccessorW result = create_zero_filled_accessor_w(
        input.shape, cpu_allocator);

    SUBCASE("elementwise_affine = true") {
      LayerNormAttrs attrs = LayerNormAttrs{
        /*axes=*/std::set<ff_dim_t>{
          ff_dim_t{1_n}, 
          ff_dim_t{2_n},
        },
        /*elementwise_affine=*/true,
        /*eps=*/1.0f,
      };

      GenericTensorAccessorR gamma =
          create_2d_accessor_r_with_contents<float>(
              {
                {1, 4, 2},
                {6, 1, 1},
                {-2, 3, 4},
                {5, 0, 1},
              },
              cpu_allocator);

      GenericTensorAccessorR beta =
          create_2d_accessor_r_with_contents<float>(

              {
                {7, -5, -3},
                {-1, 1, -1},
                {0, 2, -4},
                {3, 1, 2},
              },
              cpu_allocator);

      layer_norm_cpu_forward_kernel(
        /*attrs=*/attrs,
        /*input=*/input,
        /*output=*/result,
        /*gamma=*/gamma,
        /*beta=*/beta);

      GenericTensorAccessorR correct = create_3d_accessor_r_with_contents<float>(
          {
            {
              {7.250629901885986, -3.9974799156188965, -0.5992288589477539},
              {-1.3957313299179077, 0.6174595355987549, -0.11619961261749268},
              {0.7650808691978455, 1.8021342754364014, -9.329183578491211},
              {12.167780876159668, 1.0, 0.351118803024292},
            },
            {
              {8.90172004699707, -2.3914804458618164, -3.5701255798339844},
              {-13.956687927246094, 1.3397324085235596, -0.9726651906967163},
              {-0.8356634974479675, 2.738039255142212, -5.140251159667969},
              {4.698661804199219, 1.0, 0.7777446508407593},
            },
          },
          cpu_allocator);

      CHECK_MESSAGE(accessors_are_within_epsilon(result, correct),
                    check_kv("result=", format_accessor_w_contents(result)));
    }

    SUBCASE("elementwise_affine = false") {
      LayerNormAttrs attrs = LayerNormAttrs{
        /*axes=*/std::set<ff_dim_t>{
          ff_dim_t{1_n}, 
          ff_dim_t{2_n},
        },
        /*elementwise_affine=*/false,
        /*eps=*/1.0f,
      };

      layer_norm_cpu_forward_kernel(
        /*attrs=*/attrs,
        /*input=*/input,
        /*output=*/result,
        /*gamma=*/std::nullopt,
        /*beta=*/std::nullopt);

      GenericTensorAccessorR correct = create_3d_accessor_r_with_contents<float>(
          {
            {
              {0.2506299912929535, 0.2506299912929535, 1.200385570526123},
              {-0.06595522910356522, -0.38254043459892273, 0.8838003873825073},
              {-0.38254043459892273, -0.06595522910356522, -1.3322960138320923},
              {1.8335561752319336, -0.540833055973053, -1.648881196975708},
            },
            {
              {1.9017200469970703, 0.6521298885345459, -0.2850627303123474},
              {-2.1594479084014893, 0.3397323489189148, 0.02733481489121914},
              {0.41783174872398376, 0.2460130900144577, -0.2850627303123474},
              {0.3397323489189148, 0.02733481489121914, -1.2222553491592407},
            },
          },
          cpu_allocator);

      CHECK_MESSAGE(accessors_are_within_epsilon(result, correct),
                    check_kv("result=", format_accessor_w_contents(result)));
    }
  }
}
