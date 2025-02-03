#include "utils/vector.h"
#include "test/utils/doctest/fmt/vector.h"
#include <catch2/catch_test_macros.hpp>
#include <string>

using namespace ::FlexFlow;


  TEST_CASE("concat function") {
    SECTION("concatenates two vectors") {
      std::vector<int> v1 = {1, 2, 3};
      std::vector<int> v2 = {4, 5, 6};
      std::vector<int> result = concat(v1, v2);
      std::vector<int> expected = {1, 2, 3, 4, 5, 6};
      CHECK(result == expected);
    }

    SECTION("concatenates two string vectors") {
      std::vector<std::string> v1 = {"1", "2", "3"};
      std::vector<std::string> v2 = {"4", "5", "6"};
      std::vector<std::string> result = concat(v1, v2);
      std::vector<std::string> expected = {"1", "2", "3", "4", "5", "6"};
      CHECK(result == expected);
    }

    SECTION("concatenates multiple vectors") {
      std::vector<int> v1 = {1, 2, 3};
      std::vector<int> v2 = {4, 5, 6};
      std::vector<int> v3 = {7, 8, 9};
      std::vector<int> result = concat(v1, v2, v3);
      std::vector<int> expected = {1, 2, 3, 4, 5, 6, 7, 8, 9};
      CHECK(result == expected);
    }
  }
