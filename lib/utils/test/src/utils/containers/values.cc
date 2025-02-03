#include "utils/containers/values.h"
#include "test/utils/doctest/fmt/unordered_multiset.h"
#include <catch2/catch_test_macros.hpp>
#include <string>
#include <unordered_map>
#include <vector>

using namespace FlexFlow;


  TEST_CASE("values") {
    std::unordered_map<int, std::string> m = {
        {1, "one"}, {2, "two"}, {3, "three"}, {33, "three"}};
    std::unordered_multiset<std::string> result = values(m);
    std::unordered_multiset<std::string> correct = {
        "one", "two", "three", "three"};
    CHECK(result == correct);
  }
