#include "utils/random_utils.h"
#include "utils/containers/contains.h"
#include "utils/containers/filter.h"
#include "utils/containers/repeat.h"
#include "utils/containers/sum.h"
#include "utils/containers/zip.h"
#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace ::FlexFlow;


  TEST_CASE("select_random(std::vector<T>)") {
    std::vector<int> values = {1, 2, 3, 4, 5};

    SECTION("selected value is in container") {
      SECTION("equal weights") {
        int result = select_random(values);
        CHECK(contains(values, result));
      }

      SECTION("unequal weights") {
        std::vector<float> weights = {0.1f, 0.3f, 0.2f, 0.2f, 0.2f};
        int result = select_random(values, weights);
        CHECK(contains(values, result));
      }
    }

    SECTION("correct distribution") {
      auto check_probabilities = [](std::vector<int> const &values,
                                    std::vector<float> const &weights) {
        nonnegative_int num_iterations = 10'000_n;
        std::vector<int> trials = repeat(
            num_iterations, [&]() { return select_random(values, weights); });

        for (std::pair<int, float> const &p : zip(values, weights)) {
          int v = p.first;
          float w = p.second;
          float expectedProbability = w / sum(weights);
          int num_occurrences =
              filter(trials, [&](int c) { return (c == v); }).size();
          float observedProbability = static_cast<float>(num_occurrences) /
                                      num_iterations.unwrap_nonnegative();
          CHECK_THAT(observedProbability, Catch::Matchers::WithinAbs(expectedProbability, 0.02));
        }
      };

      SECTION("equal weights") {
        std::vector<float> weights = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        check_probabilities(values, weights);
      }

      SECTION("unequal weights") {
        std::vector<float> weights = {0.1f, 0.2f, 0.3f, 0.2f, 0.2f};
        check_probabilities(values, weights);
      }
    }
  }
