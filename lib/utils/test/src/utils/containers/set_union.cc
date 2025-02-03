#include "utils/containers/set_union.h"
#include "test/utils/doctest/fmt/unordered_set.h"
#include <catch2/catch_test_macros.hpp>
#include <unordered_set>

using namespace FlexFlow;


  TEST_CASE("set_union") {
    std::unordered_set<int> s1 = {1, 2, 3};
    std::unordered_set<int> s2 = {2, 3, 4};
    std::unordered_set<int> result = set_union(s1, s2);
    std::unordered_set<int> correct = {1, 2, 3, 4};
    CHECK(result == correct);
  }
