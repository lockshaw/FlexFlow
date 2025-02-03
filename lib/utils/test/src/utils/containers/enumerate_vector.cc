#include "utils/containers/enumerate_vector.h"
#include "test/utils/doctest/fmt/map.h"
#include <catch2/catch_test_macros.hpp>

using namespace ::FlexFlow;


  TEST_CASE("enumerate_vector(std::vector<T>)") {
    SECTION("input vector is empty") {
      std::vector<int> input = {};

      std::map<nonnegative_int, int> result = enumerate_vector(input);
      std::map<nonnegative_int, int> correct = {};

      CHECK(result == correct);
    }

    SECTION("input vector is not empty") {
      std::vector<int> input = {2, 3, 1, 3, 3};

      std::map<nonnegative_int, int> result = enumerate_vector(input);
      std::map<nonnegative_int, int> correct = {
          {0_n, 2},
          {1_n, 3},
          {2_n, 1},
          {3_n, 3},
          {4_n, 3},
      };

      CHECK(result == correct);
    }
  }
