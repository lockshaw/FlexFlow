#include "utils/containers/reversed.h"
#include "test/utils/doctest/fmt/vector.h"
#include <catch2/catch_test_macros.hpp>

using namespace ::FlexFlow;


  TEST_CASE("reversed(std::vector<T>)") {
    SECTION("non-empty input") {
      std::vector<int> input = {1, 2, 3, 2};

      std::vector<int> result = reversed(input);
      std::vector<int> correct = {2, 3, 2, 1};

      CHECK(result == correct);
    }

    SECTION("empty input") {
      std::vector<int> input = {};

      std::vector<int> result = reversed(input);
      std::vector<int> correct = {};

      CHECK(result == correct);
    }
  }
