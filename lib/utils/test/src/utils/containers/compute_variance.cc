#include <doctest/doctest.h>
#include "utils/containers/compute_variance.h"
#include <vector>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("compute_variance") {
    SUBCASE("input is empty") {
      std::vector<float> input = {};

      CHECK_THROWS(compute_variance(input));
    }

    SUBCASE("input has one element") {
      std::vector<float> input = {3.8f};

      float result = compute_variance(input);
      float correct = 0.0;

      CHECK(result == correct);
    }

    SUBCASE("input has all the same element") {
      std::vector<float> input = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f};

      float result = compute_variance(input);
      float correct = 0.0;

      CHECK(result == correct);
    }

    SUBCASE("input has different elements") {
      std::vector<float> input = {1.0f, 3.0f, 5.0f};

      float result = compute_variance(input);
      float correct = (4.0f + 0.0f + 4.0f) / 3;

      CHECK(result == correct);
    }

    SUBCASE("input takes repeats into account") {
      std::vector<float> input = {1.0f, 3.0f, 5.0f, 1.0f, 5.0f};

      float result = compute_variance(input);
      float correct = (4.0f + 0.0f + 4.0f + 4.0f + 4.0f) / 5;

      CHECK(result == correct);
    }
  }
}
