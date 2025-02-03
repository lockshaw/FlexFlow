#include "utils/containers/map_keys.h"
#include "test/utils/doctest/fmt/unordered_map.h"
#include <catch2/catch_test_macros.hpp>
#include <string>
#include <unordered_map>

using namespace FlexFlow;


  TEST_CASE("map_keys") {
    SECTION("Distinct keys after transformation") {
      std::unordered_map<int, std::string> m = {{1, "one"}, {2, "two"}};
      auto f = [](int x) { return x * x; };
      std::unordered_map<int, std::string> result = map_keys(m, f);
      std::unordered_map<int, std::string> correct = {{1, "one"}, {4, "two"}};
      CHECK(correct == result);
    }

    SECTION("Non-distinct keys after transformation") {
      std::unordered_map<int, std::string> m = {
          {1, "one"}, {2, "two"}, {-1, "minus one"}};
      auto f = [](int x) { return std::abs(x); };
      CHECK_THROWS(map_keys(m, f));
    }
  }
