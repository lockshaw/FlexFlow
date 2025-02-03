#include "utils/containers/lookup_in_map.h"
#include <catch2/catch_test_macros.hpp>
#include <functional>
#include <string>

using namespace FlexFlow;



  TEST_CASE("lookup_in_map") {

    std::unordered_map<std::string, int> map = {{"a", 1}, {"b", 2}};

    SECTION("existing keys") {
      std::function<int(std::string const &)> func = lookup_in_map(map);
      CHECK(func("a") == 1);
      CHECK(func("b") == 2);
    }

    SECTION("missing key") {
      std::function<int(std::string const &)> func = lookup_in_map(map);
      CHECK_THROWS(func("c"));
    }

    SECTION("empty map") {
      std::unordered_map<std::string, int> map = {};
      std::function<int(std::string const &)> func = lookup_in_map(map);
      CHECK_THROWS(func("a"));
    }
  }
