#include "utils/fmt/set.h"
#include <catch2/catch_test_macros.hpp>

using namespace ::FlexFlow;


  TEST_CASE("fmt::to_string(std::set<int>)") {
    std::set<int> input = {0, 1, 3, 2};
    std::string result = fmt::to_string(input);
    std::string correct = "{0, 1, 2, 3}";
    CHECK(result == correct);
  }
