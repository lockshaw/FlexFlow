#include "utils/containers/get_element_counts.h"
#include "test/utils/doctest/fmt/unordered_map.h"
#include <catch2/catch_test_macros.hpp>

using namespace ::FlexFlow;


  TEST_CASE("get_element_counts") {
    std::vector<int> input = {1, 2, 3, 2, 3, 3, 2, 3};
    std::unordered_map<int, int> result = get_element_counts(input);
    std::unordered_map<int, int> correct = {{1, 1}, {2, 3}, {3, 4}};
    CHECK(result == correct);
  }
