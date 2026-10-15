#include "utils/binary_relation/binary_relation_transform_l_and_r.h"
#include <doctest/doctest.h>
#include <libassert/assert.hpp>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("binary_relation_transform_l_and_r") {
    SUBCASE("relation is empty") {
      BinaryRelation<int, std::string> b = BinaryRelation<int, std::string>{};

      BinaryRelation<std::string, float> result =
          binary_relation_transform_l_and_r(
              b,
              [&](int) -> std::string { PANIC(); },
              [&](std::string) -> float { PANIC(); });

      BinaryRelation<std::string, float> correct =
          BinaryRelation<std::string, float>{};

      CHECK(result == correct);
    }

    SUBCASE("relation is not empty") {
      BinaryRelation<int, std::string> b = BinaryRelation<int, std::string>{
          {1, "one"},
          {2, "two"},
          {3, "three"},
      };

      BinaryRelation<std::string, int> result =
          binary_relation_transform_l_and_r(
              b,
              [&](int l) -> std::string { return std::to_string(l); },
              [&](std::string r) -> int { return r.size(); });

      BinaryRelation<std::string, int> correct =
          BinaryRelation<std::string, int>{
              {"1", 3},
              {"2", 3},
              {"3", 5},
          };

      CHECK(result == correct);
    }

    SUBCASE("produces duplicate outputs") {
      BinaryRelation<int, std::string> b = BinaryRelation<int, std::string>{
          {1, "one"},
          {2, "two"},
          {3, "three"},
      };

      BinaryRelation<std::string, int> result =
          binary_relation_transform_l_and_r(
              b,
              [&](int l) -> std::string { return "a"; },
              [&](std::string r) -> int { return 1; });

      BinaryRelation<std::string, int> correct =
          BinaryRelation<std::string, int>{
              {"a", 1},
          };

      CHECK(result == correct);
    }
  }
}
