#include "utils/containers/sum.h"
#include <catch2/catch_test_macros.hpp>
#include <vector>

using namespace ::FlexFlow;


  TEST_CASE("sum(std::vector<int>)") {
    SECTION("input is empty") {
      std::vector<int> input = {};

      int result = sum(input);
      int correct = 0;

      CHECK(result == correct);
    }

    SECTION("input is not empty") {
      std::vector<int> input = {1, 3, 2};

      int result = sum(input);
      int correct = 6;

      CHECK(result == correct);
    }
  }
