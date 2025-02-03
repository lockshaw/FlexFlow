#include "utils/containers/get_only.h"
#include <catch2/catch_test_macros.hpp>

using namespace ::FlexFlow;


  TEST_CASE("get_only(std::vector<int>)") {
    std::vector<int> input = {5};
    int result = get_only(input);
    int correct = 5;
    CHECK(result == correct);
  }

  TEST_CASE("get_only(std::unordered_set<int>)") {
    std::unordered_set<int> input = {5};
    int result = get_only(input);
    int correct = 5;
    CHECK(result == correct);
  }
