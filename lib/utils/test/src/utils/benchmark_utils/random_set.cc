#include "utils/benchmark_utils/random_set.h"
#include "test/utils/doctest/fmt/set.h"
#include "utils/benchmark_utils/random_int.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("random_set") {
    SUBCASE("is fixed over time") {
      std::mt19937 gen1;

      gen1.seed(5);

      std::set<int> result = random_set(gen1, 3_n, random_int);
      std::set<int> correct = {118498407, 476726705, 1869883383};

      CHECK(result == correct);
    }

    SUBCASE("returns the correct size") {
      std::mt19937 gen1;

      gen1.seed(5);

      std::set<int> result = random_set(gen1, 6_n, random_int);

      CHECK(result.size() == 6);
    }

    SUBCASE("is deterministic with respect to gen") {
      std::mt19937 gen1;

      gen1.seed(5);

      SUBCASE("same gens produce the same value") {
        std::mt19937 gen2;
        gen2.seed(5);

        std::set<int> result1 = random_set(gen1, 10_n, random_int);
        std::set<int> result2 = random_set(gen2, 10_n, random_int);

        CHECK(result1 == result2);
      }

      SUBCASE("different gens produce different values") {
        std::mt19937 gen2;
        gen2.seed(8);

        std::set<int> result1 = random_set(gen1, 10_n, random_int);
        std::set<int> result2 = random_set(gen2, 10_n, random_int);

        CHECK(result1 != result2);
      }

      SUBCASE("repeated use of the same gen produces different values") {
        std::set<int> result1 = random_set(gen1, 10_n, random_int);
        std::set<int> result2 = random_set(gen1, 10_n, random_int);

        CHECK(result1 != result2);
      }
    }
  }
}
