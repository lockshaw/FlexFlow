#include "utils/containers/repeat_element.h"
#include "test/utils/doctest/fmt/unordered_set.h"
#include "test/utils/doctest/fmt/vector.h"
#include <catch2/catch_test_macros.hpp>
#include <unordered_set>

using namespace FlexFlow;


  TEST_CASE("repeat_element") {
    SECTION("ints") {
      int x = 42;
      std::vector<int> result = repeat_element(nonnegative_int{5}, x);
      std::vector<int> correct = {42, 42, 42, 42, 42};
      CHECK(result == correct);
    }
    SECTION("unordered_set") {
      std::unordered_set<float> x = {1.0, 1.5};
      std::vector<std::unordered_set<float>> result =
          repeat_element(nonnegative_int{3}, x);
      std::vector<std::unordered_set<float>> correct = {
          {1.0, 1.5}, {1.0, 1.5}, {1.0, 1.5}};
      CHECK(result == correct);
    }
  }
