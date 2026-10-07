#include "utils/benchmark_utils/random_subset.h"
#include "test/utils/doctest/fmt/set.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("random_subset") {
    std::set<int> superset = {1, 3, 4, 5, 6, 7, 8, 10, 12, 15, 17, 22};

    SUBCASE("is fixed over time") {
      std::mt19937 gen1;
      gen1.seed(5);

      std::set<int> result = random_subset(gen1, 3_n, superset);
      std::set<int> correct = {1, 4, 17};

      CHECK(result == correct);
    }

    SUBCASE("returns the correct size") {
      std::mt19937 gen1;
      gen1.seed(5);

      std::set<int> result = random_subset(gen1, 4_n, superset);

      CHECK(result.size() == 4);
    }

    SUBCASE("is deterministic with respect to gen") {
      std::mt19937 gen1;

      gen1.seed(5);

      SUBCASE("same gens produce the same value") {
        std::mt19937 gen2;
        gen2.seed(5);

        std::set<int> result1 = random_subset(gen1, 4_n, superset);
        std::set<int> result2 = random_subset(gen2, 4_n, superset);

        CHECK(result1 == result2);
      }

      SUBCASE("different gens produce different values") {
        std::mt19937 gen2;
        gen2.seed(8);

        std::set<int> result1 = random_subset(gen1, 4_n, superset);
        std::set<int> result2 = random_subset(gen2, 4_n, superset);

        CHECK(result1 != result2);
      }

      SUBCASE("repeated use of the same gen produces different values") {
        std::set<int> result1 = random_subset(gen1, 4_n, superset);
        std::set<int> result2 = random_subset(gen1, 4_n, superset);

        CHECK(result1 != result2);
      }
    }
  }
}
