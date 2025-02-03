#include "utils/containers/maximum.h"
#include "test/utils/doctest/fmt/vector.h"
#include <catch2/catch_test_macros.hpp>
#include <vector>

using namespace ::FlexFlow;



  TEST_CASE("maximum") {

    SECTION("non-empty container") {
      std::vector<int> input = {1, 5, 3, 4, 2};
      int correct = 5;
      int result = maximum(input);
      CHECK(correct == result);
    }

    SECTION("empty container") {
      std::vector<int> input = {};

      CHECK_THROWS(maximum(input));
    }
  }
