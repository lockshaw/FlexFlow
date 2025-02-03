#include "utils/containers/are_disjoint.h"
#include <catch2/catch_test_macros.hpp>
#include <unordered_set>

using namespace FlexFlow;


  TEST_CASE("are_disjoint") {
    SECTION("disjoint") {
      std::unordered_set<int> l = {1, 2, 3};
      std::unordered_set<int> r = {4, 5, 6};
      CHECK(are_disjoint(l, r));
    }
    SECTION("not disjoint") {
      std::unordered_set<int> l = {1, 2, 3, 4};
      std::unordered_set<int> r = {3, 4, 5, 6};
      CHECK_FALSE(are_disjoint(l, r));
    }

    SECTION("one empty set") {
      std::unordered_set<int> l = {1, 2};
      std::unordered_set<int> r = {};
      CHECK(are_disjoint(l, r));
    }
    SECTION("both empty sets") {
      std::unordered_set<int> l = {};
      std::unordered_set<int> r = {};
      CHECK(are_disjoint(l, r));
    }
  }
