#include "op-attrs/relative_ff_dim_t.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;


  TEST_CASE("ff_dim_t_from_relative_ff_dim_t") {
    nonnegative_int input_dim = 5_n;

    SUBCASE("relative index is zero") {
      relative_ff_dim_t relative_ff_dim = relative_ff_dim_t{0};
      ff_dim_t ff_dim =
          ff_dim_t_from_relative_ff_dim_t(relative_ff_dim, input_dim);
      CHECK(ff_dim == ff_dim_t{0_n});
    }

    SUBCASE("relative index is positive") {

      SUBCASE("relative index is in range") {
        relative_ff_dim_t relative_ff_dim = relative_ff_dim_t{1};
        ff_dim_t ff_dim =
            ff_dim_t_from_relative_ff_dim_t(relative_ff_dim, input_dim);
        CHECK(ff_dim == ff_dim_t{1_n});
      }

      SUBCASE("relative index is out of range") {
        relative_ff_dim_t relative_ff_dim = relative_ff_dim_t{10};
        ff_dim_t ff_dim =
            ff_dim_t_from_relative_ff_dim_t(relative_ff_dim, input_dim);
        CHECK(ff_dim == ff_dim_t{10_n});
      }
    }

    SUBCASE("relative index is negative") {

      SUBCASE("relative index is in range") {
        relative_ff_dim_t relative_ff_dim = relative_ff_dim_t{-1};
        ff_dim_t ff_dim =
            ff_dim_t_from_relative_ff_dim_t(relative_ff_dim, input_dim);
        CHECK(ff_dim == ff_dim_t{4_n});
      }

      SUBCASE("relative index is out of range") {
        relative_ff_dim_t relative_ff_dim = relative_ff_dim_t{-10};
        CHECK_THROWS(
            ff_dim_t_from_relative_ff_dim_t(relative_ff_dim, input_dim));
      }
    }
  }
}
