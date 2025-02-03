#include "utils/containers/any_of.h"
#include <catch2/catch_test_macros.hpp>
#include "utils/fmt/vector.h"
#include <vector>

using namespace ::FlexFlow;


  TEST_CASE("any_of(C, F)") {
    SECTION("has element matching condition") {
      std::vector<int> input = {1, 2, 3};

      bool result = any_of(input, [](int x) { return x > 1; });
      bool correct = true;

      CHECK(result == correct);
    }

    SECTION("does not have element matching condition") {
      std::vector<int> input = {1, 2, 3};

      bool result = any_of(input, [](int x) { return x > 5; });
      bool correct = false;

      CHECK(result == correct);
    }

    SECTION("input is empty") {
      std::vector<int> input = {};

      bool result = any_of(input, [](int x) { return true; });
      bool correct = false;

      CHECK(result == correct);
    }
  }
