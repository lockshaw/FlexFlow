#include "utils/containers/map_values.h"
#include "test/utils/doctest/fmt/unordered_map.h"
#include <catch2/catch_test_macros.hpp>
#include <string>
#include <unordered_map>

using namespace FlexFlow;


  TEST_CASE("map_values") {
    std::unordered_map<int, std::string> m = {{1, "one"}, {3, "three"}};
    auto f = [](std::string const &s) { return s.size(); };
    std::unordered_map<int, size_t> result = map_values(m, f);
    std::unordered_map<int, size_t> correct = {{1, 3}, {3, 5}};
    CHECK(result == correct);
  }
