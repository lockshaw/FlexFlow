#include "utils/benchmark_utils/random_int.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("random_int") {
    SUBCASE("is fixed over time") {
      std::mt19937 gen1;
      gen1.seed(5);

      int result = random_int(gen1);
      int correct = 476726705;

      CHECK(result == correct);
    }

    SUBCASE("is deterministic with respect to gen") {
      std::mt19937 gen1;
      gen1.seed(5);

      SUBCASE("same gens produce the same value") {
        std::mt19937 gen2;
        gen2.seed(5);

        int result1 = random_int(gen1);
        int result2 = random_int(gen2);

        CHECK(result1 == result2);
      }

      SUBCASE("different gens produce different values") {
        std::mt19937 gen2;
        gen2.seed(8);

        int result1 = random_int(gen1);
        int result2 = random_int(gen2);

        CHECK(result1 != result2);
      }

      SUBCASE("repeated use of the same gen produces different values") {
        int result1 = random_int(gen1);
        int result2 = random_int(gen1);

        CHECK(result1 != result2);
      }
    }
  }
}
