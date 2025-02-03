#include "utils/containers/are_all_same.h"
#include <fmt/format.h>
#include <catch2/catch_test_macros.hpp>
#include <vector>

using namespace ::FlexFlow;


  TEST_CASE("are_all_same(std::vector<T>)") {
    SECTION("input is empty") {
      std::vector<int> input = {};

      bool result = are_all_same(input);
      bool correct = true;

      CHECK(result == correct);
    }

    SECTION("input elements are all same") {
      std::vector<int> input = {1, 1, 1};

      bool result = are_all_same(input);
      bool correct = true;

      CHECK(result == correct);
    }

    SECTION("input elements are not all same") {
      std::vector<int> input = {1, 1, 2, 1};

      bool result = are_all_same(input);
      bool correct = false;

      CHECK(result == correct);
    }
  }
