#include "utils/containers/get_one_of.h"
#include "utils/containers/contains.h"
#include <catch2/catch_test_macros.hpp>
#include <unordered_set>
using namespace FlexFlow;


  TEST_CASE("get_one_of") {
    SECTION("non-empty set") {
      std::unordered_set<int> s = {1, 2, 3};
      CHECK(contains(s, get_one_of(s)));
    }

    SECTION("empty set") {
      std::unordered_set<int> s = {};
      CHECK_THROWS(get_one_of(s));
    }
  }
