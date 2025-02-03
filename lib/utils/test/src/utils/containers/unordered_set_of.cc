#include "utils/containers/unordered_set_of.h"
#include "test/utils/doctest/fmt/unordered_set.h"
#include <catch2/catch_test_macros.hpp>
#include <vector>

using namespace ::FlexFlow;


  TEST_CASE("unordered_set_of") {
    std::vector<int> input = {1, 2, 3, 3, 2, 3};
    std::unordered_set<int> result = unordered_set_of(input);
    std::unordered_set<int> correct = {1, 2, 3};
    CHECK(result == correct);
  }
