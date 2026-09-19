#include <doctest/doctest.h>
#include "op-attrs/ff_ordered/ff_ordered_restrict_dims.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("ff_ordered_restrict_dims") {
    FFOrdered<int> input = {
      1,
      4,
      3,
    };

    std::set<ff_dim_t> dim_set = {
      ff_dim_t{0_n},
      ff_dim_t{2_n},
      ff_dim_t{4_n},
    };

    FFOrdered<int> result = ff_ordered_restrict_dims(input, dim_set);
    FFOrdered<int> correct = {
      1,
      3,
    };

    CHECK(result == correct);
  }
}
