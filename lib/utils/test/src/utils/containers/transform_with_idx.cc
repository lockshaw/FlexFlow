#include <doctest/doctest.h>
#include "utils/containers/transform_with_idx.h"
#include "test/utils/doctest/fmt/vector.h"
#include "test/utils/doctest/fmt/pair.h"
#include <libassert/assert.hpp>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("transform_with_idx") {
    SUBCASE("input is empty") {
      std::vector<int> input = {};

      std::vector<std::string> result =
        transform_with_idx(input,
                           [&](nonnegative_int idx, int val) -> std::string {
                             PANIC();
                           });

      std::vector<std::string> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("input is not empty") {
      std::vector<int> input = {1, 3, 2, 3};

      auto f = [](nonnegative_int idx, int val)
        -> std::pair<nonnegative_int, int>
      {
        return std::pair{idx, val};
      };

      std::vector<std::pair<nonnegative_int, int>> result =
        transform_with_idx(input, f);

      std::vector<std::pair<nonnegative_int, int>> correct = {
        {0_n, 1},
        {1_n, 3},
        {2_n, 2},
        {3_n, 3},
      };

      CHECK(result == correct);
    }
  }
}
