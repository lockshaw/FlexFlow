#include "utils/bidict/try_merge_nondisjoint_bidicts.h"
#include <catch2/catch_test_macros.hpp>

using namespace ::FlexFlow;


  TEST_CASE("try_merge_nondisjoint_bidicts(bidict<L, R>, bidict<L, R>)") {
    bidict<int, std::string> d1 = {
        {0, "zero"},
        {1, "one"},
    };
    bidict<int, std::string> d2 = {
        {0, "zero"},
        {2, "two"},
    };

    SECTION("compatible neither superset") {
      std::optional<bidict<int, std::string>> result =
          try_merge_nondisjoint_bidicts(d1, d2);
      std::optional<bidict<int, std::string>> correct = {{
          {0, "zero"},
          {1, "one"},
          {2, "two"},
      }};
      CHECK(result == correct);
    }

    SECTION("mismatched key") {
      d1.equate(2, "three");
      std::optional<bidict<int, std::string>> result =
          try_merge_nondisjoint_bidicts(d1, d2);
      std::optional<bidict<int, std::string>> correct = std::nullopt;
      CHECK(result == correct);
    }

    SECTION("repeated value") {
      d1.equate(3, "one");
      std::optional<bidict<int, std::string>> result =
          try_merge_nondisjoint_bidicts(d1, d2);
      std::optional<bidict<int, std::string>> correct = std::nullopt;
      CHECK(result == correct);
    }

    SECTION("left superset") {
      d1.equate(2, "two");
      std::optional<bidict<int, std::string>> result =
          try_merge_nondisjoint_bidicts(d1, d2);
      std::optional<bidict<int, std::string>> correct = d1;
      CHECK(result == correct);
    }

    SECTION("right superset") {
      d2.equate(1, "one");
      std::optional<bidict<int, std::string>> result =
          try_merge_nondisjoint_bidicts(d1, d2);
      std::optional<bidict<int, std::string>> correct = d2;
      CHECK(result == correct);
    }

    SECTION("equal") {
      d1.equate(2, "two");
      d2.equate(1, "one");
      std::optional<bidict<int, std::string>> result =
          try_merge_nondisjoint_bidicts(d1, d2);
      std::optional<bidict<int, std::string>> correct = d1;
      CHECK(result == correct);
    }
  }
