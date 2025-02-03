#include "utils/containers/filter_keys.h"
#include "test/utils/doctest/fmt/unordered_map.h"
#include <catch2/catch_test_macros.hpp>
#include <string>
#include <unordered_map>

using namespace FlexFlow;


  TEST_CASE("filter_keys") {
    std::unordered_map<int, std::string> m = {
        {1, "one"}, {2, "two"}, {3, "three"}};
    auto f = [](int x) { return x % 2 == 1; };
    std::unordered_map<int, std::string> result = filter_keys(m, f);
    std::unordered_map<int, std::string> correct = {{1, "one"}, {3, "three"}};
    CHECK(result == correct);
  }
