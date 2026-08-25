#include <doctest/doctest.h>
#include "utils/binary_relation/binary_relation_is_left_k_unique.h"
#include "test/utils/doctest/fmt/optional.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("binary_relation_is_left_k_unique") {
    std::string s_a = "a";
    std::string s_b = "b";
    std::string s_c = "c";
    std::string s_d = "d";
    std::string s_e = "e";

    SUBCASE("binary relation is right 3-unique") {
      BinaryRelation<int, std::string> rel = BinaryRelation<int, std::string>{
        {1, s_a},
        {2, s_a},
        {3, s_a},
        {3, s_b},
        {4, s_b},
        {5, s_b},
      };

      std::optional<nonnegative_int> result = binary_relation_is_left_k_unique(rel);

      std::optional<nonnegative_int> correct = 3_n;

      CHECK(result == correct);
    }

    SUBCASE("binary relation is right 1-unique") {
      BinaryRelation<int, std::string> rel = {
        {1, s_a},
        {2, s_b},
        {3, s_c},
      };

      std::optional<nonnegative_int> result = binary_relation_is_left_k_unique(rel);

      std::optional<nonnegative_int> correct = 1_n;

      CHECK(result == correct);
    }

    SUBCASE("binary relation is universal") {
      BinaryRelation<int, std::string> rel = {
        {1, s_a},
        {1, s_b},
        {2, s_a},
        {2, s_b},
      };

      std::optional<nonnegative_int> result = binary_relation_is_left_k_unique(rel);

      std::optional<nonnegative_int> correct = 2_n;

      CHECK(result == correct);
    }

    SUBCASE("binary relation is empty") {
      BinaryRelation<int, std::string> rel = {};

      std::optional<nonnegative_int> result = binary_relation_is_left_k_unique(rel);

      std::optional<nonnegative_int> correct = 0_n;

      CHECK(result == correct);
    }

    SUBCASE("binary relation not right k-unique due to inconsistent k size") {
      BinaryRelation<int, std::string> rel = {
        {1, s_a},
        {2, s_a},
        {3, s_a},
        {4, s_b},
        {5, s_b},
      };

      std::optional<nonnegative_int> result = binary_relation_is_left_k_unique(rel);

      std::optional<nonnegative_int> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("binary relation is right k-unique") {
      BinaryRelation<int, std::string> rel = {
        {1, s_a},
        {1, s_b},
        {1, s_c},
        {2, s_c},
        {2, s_d},
        {2, s_e},
      };

      std::optional<nonnegative_int> result = binary_relation_is_left_k_unique(rel);

      std::optional<nonnegative_int> correct = std::nullopt;

      CHECK(result == correct);
    }
  }
}
