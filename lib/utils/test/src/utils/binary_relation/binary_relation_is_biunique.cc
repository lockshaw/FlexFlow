#include <doctest/doctest.h>
#include "utils/binary_relation/binary_relation_is_biunique.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("binary_relation_is_biunique") {
    std::string s_a = "a";
    std::string s_b = "b";
    std::string s_c = "c";

    SUBCASE("relation is biunique") {
      BinaryRelation<int, std::string> rel = BinaryRelation<int, std::string>{
        {1, s_a},
        {2, s_b},
        {3, s_c},
      };

      CHECK(binary_relation_is_biunique(rel));
    }

    SUBCASE("relation is empty") {
      BinaryRelation<int, std::string> rel = BinaryRelation<int, std::string>{};

      CHECK(binary_relation_is_biunique(rel));
    }

    SUBCASE("relation is only left-unique") {
      BinaryRelation<int, std::string> rel = BinaryRelation<int, std::string>{
        {1, s_a},
        {2, s_a},
        {3, s_c},
      };

      CHECK_FALSE(binary_relation_is_biunique(rel));
    }

    SUBCASE("relation is only right-unique") {
      BinaryRelation<int, std::string> rel = BinaryRelation<int, std::string>{
        {1, s_a},
        {2, s_b},
        {2, s_c},
      };

      CHECK_FALSE(binary_relation_is_biunique(rel));
    }
  }
}
