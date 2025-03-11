#include "kernels/accessor.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("calculate_accessor_offset") {
    ArrayShape shape = ArrayShape{LegionOrdered<nonnegative_int>{
      10_n,
    }};

    SUBCASE("num indices does not match num dims") {
      NOT_IMPLEMENTED();
    }

    SUBCASE("index out of bounds") {
      NOT_IMPLEMENTED();
    }

    SUBCASE("index in bounds") {
      NOT_IMPLEMENTED();
    }

    SUBCASE("num dims is zero") {
      NOT_IMPLEMENTED();
    }

    SUBCASE("num dims is zero") {

    }
  }
}
