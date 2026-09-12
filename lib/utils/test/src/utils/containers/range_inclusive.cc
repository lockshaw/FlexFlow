#include <doctest/doctest.h>
#include "test/utils/doctest/fmt/vector.h"
#include "utils/containers/range_inclusive.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("range_inclusive") {
    SUBCASE("step=1") {
      std::vector<int> result = range_inclusive(0, 5);
      std::vector<int> correct = {0, 1, 2, 3, 4, 5};
      CHECK(result == correct);
    }

    SUBCASE("step = 2") {
      std::vector<int> result = range_inclusive(-2, 10, 2);
      std::vector<int> correct = {-2, 0, 2, 4, 6, 8, 10};
      CHECK(result == correct);
    }

    SUBCASE("step = -1") {
      std::vector<int> result = range_inclusive(5, 0, -1);
      std::vector<int> correct = {5, 4, 3, 2, 1, 0};
      CHECK(result == correct);
    }

    SUBCASE("single argument") {
      std::vector<int> result = range_inclusive(5);
      std::vector<int> correct = {0, 1, 2, 3, 4, 5};
      CHECK(result == correct);
    }

    SUBCASE("start = end") {
      std::vector<int> result = range_inclusive(5, 5);
      std::vector<int> correct = {5};
      CHECK(result == correct);
    }

    SUBCASE("start > end") {
      std::vector<int> result = range_inclusive(5, 4);
      std::vector<int> correct = {};
      CHECK(result == correct);
    }

    SUBCASE("start < end, step < 0") {
      std::vector<int> result = range_inclusive(0, 10, -1);
      std::vector<int> correct = {};
      CHECK(result == correct);
    }

    SUBCASE("step = 0") {
      SUBCASE("output is nonempty") {
        CHECK_THROWS(range_inclusive(2, 5, 0));
      }

      SUBCASE("output is empty") {
        CHECK_THROWS(range_inclusive(3, 3, 0));
      }
    }
  }
}
