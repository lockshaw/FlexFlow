#include "utils/benchmark_utils/random_hex_string.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("random_hex_string") {
    SUBCASE("is fixed over time") {
      std::mt19937 gen1;
      gen1.seed(5);

      std::string result = random_hex_string(gen1, 6_n);
      std::string correct = "30dd35";

      CHECK(result == correct);
    }

    SUBCASE("returns the correct size") {
      std::mt19937 gen1;

      gen1.seed(5);

      std::string result = random_hex_string(gen1, 6_n);

      CHECK(result.size() == 6);
    }

    SUBCASE("is deterministic with respect to gen") {
      std::mt19937 gen1;

      gen1.seed(5);

      SUBCASE("same gens produce the same value") {
        std::mt19937 gen2;
        gen2.seed(5);

        std::string result1 = random_hex_string(gen1, 10_n);
        std::string result2 = random_hex_string(gen2, 10_n);

        CHECK(result1 == result2);
      }

      SUBCASE("different gens produce different values") {
        std::mt19937 gen2;
        gen2.seed(8);

        std::string result1 = random_hex_string(gen1, 10_n);
        std::string result2 = random_hex_string(gen2, 10_n);

        CHECK(result1 != result2);
      }

      SUBCASE("repeated use of the same gen produces different values") {
        std::string result1 = random_hex_string(gen1, 10_n);
        std::string result2 = random_hex_string(gen1, 10_n);

        CHECK(result1 != result2);
      }
    }
  }
}
