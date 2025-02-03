#include "utils/containers/are_all_distinct.h"
#include <catch2/catch_test_macros.hpp>
#include <vector>

using namespace ::FlexFlow;


  TEST_CASE("are_all_distinct") {

    SECTION("Empty Container") {
      std::vector<int> input = {};
      CHECK(are_all_distinct(input));
    }
    SECTION("All elements are distinct") {
      std::vector<int> input = {1, 2, 3, 4};
      CHECK(are_all_distinct(input));
    }

    SECTION("Not all elements are distinct") {
      std::vector<int> input = {2, 2, 3, 4};
      CHECK_FALSE(are_all_distinct(input));
    }
  }
