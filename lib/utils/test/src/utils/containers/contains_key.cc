#include "utils/containers/contains_key.h"
#include <catch2/catch_test_macros.hpp>
#include <map>
#include <string>
#include <unordered_map>

using namespace ::FlexFlow;


  TEST_CASE("contains_key(std::unordered_map<K, V>, K)") {
    std::unordered_map<int, std::string> m = {
        {1, "one"},
    };
    CHECK(contains_key(m, 1));
    CHECK_FALSE(contains_key(m, 2));
  }

  TEST_CASE("contains_key(std::map<K, V>, K)") {
    std::map<int, std::string> m = {
        {1, "one"},
    };
    CHECK(contains_key(m, 1));
    CHECK_FALSE(contains_key(m, 2));
  }
