#include "utils/containers/is_superseteq_of.h"
#include <catch2/catch_test_macros.hpp>
#include <unordered_set>

using namespace ::FlexFlow;


  TEST_CASE("is_superseteq_of") {
    std::unordered_set<int> super = {1, 2, 3, 4};

    SECTION("true containment") {
      std::unordered_set<int> sub = {1, 2, 3};
      CHECK(is_superseteq_of(super, sub));
    }

    SECTION("false containment") {
      std::unordered_set<int> sub = {1, 2, 5};
      CHECK_FALSE(is_superseteq_of(super, sub));
    }

    SECTION("reflexive") {
      CHECK(is_superseteq_of(super, super));
    }
  }
