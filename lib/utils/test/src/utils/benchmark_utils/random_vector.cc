#include "utils/benchmark_utils/random_vector.h"
#include "test/utils/doctest/fmt/vector.h"
#include "utils/benchmark_utils/random_int.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("random_vector") {
    SUBCASE("returns the correct size") {
      std::mt19937 gen1;

      gen1.seed(5);

      std::vector<int> result = random_vector(gen1, 6_n, random_int);

      CHECK(result.size() == 6);
    }

    SUBCASE("is deterministic with respect to gen") {
      std::mt19937 gen1;

      gen1.seed(5);

      SUBCASE("same gens produce the same value") {
        std::mt19937 gen2;
        gen2.seed(5);

        std::vector<int> result1 = random_vector(gen1, 10_n, random_int);
        std::vector<int> result2 = random_vector(gen2, 10_n, random_int);

        CHECK(result1 == result2);
      }

      SUBCASE("different gens produce different values") {
        std::mt19937 gen2;
        gen2.seed(8);

        std::vector<int> result1 = random_vector(gen1, 10_n, random_int);
        std::vector<int> result2 = random_vector(gen2, 10_n, random_int);

        CHECK(result1 != result2);
      }

      SUBCASE("repeated use of the same gen produces different values") {
        std::vector<int> result1 = random_vector(gen1, 10_n, random_int);
        std::vector<int> result2 = random_vector(gen1, 10_n, random_int);

        CHECK(result1 != result2);
      }
    }
  }
}
