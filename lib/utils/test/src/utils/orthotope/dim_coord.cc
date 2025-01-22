#include "utils/orthotope/dim_coord.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("flatten_coord") {
    DimCoord<int> coord = DimCoord<int>{{
      {3, 4_n},
      {7, 0_n},
      {1, 1_n},
    }};

    Orthotope<int> domain = Orthotope<int>{{
      {3, 5_n},
      {7, 2_n},
      {1, 3_n},
    }};

    nonnegative_int result = flatten_coord(coord, domain);
    nonnegative_int correct = nonnegative_int{1 * 2 * 5 + 4 * 2 + 0};

    CHECK(result == correct);
  }
}
