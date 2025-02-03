#include "utils/containers/try_merge_nondisjoint_unordered_maps.h"
#include "test/utils/doctest/fmt/optional.h"
#include "test/utils/doctest/fmt/unordered_map.h"
#include <catch2/catch_test_macros.hpp>

using namespace ::FlexFlow;


  TEST_CASE("try_merge_nondisjoing_unordered_maps(std::unordered_map<K, V>, "
            "std::unordered_map<K, V>)") {
    std::unordered_map<int, std::string> d1 = {
        {0, "zero"},
        {1, "one"},
    };
    std::unordered_map<int, std::string> d2 = {
        {0, "zero"},
        {2, "two"},
    };

    SECTION("compatible neither superset") {
      std::optional<std::unordered_map<int, std::string>> result =
          try_merge_nondisjoint_unordered_maps(d1, d2);
      std::optional<std::unordered_map<int, std::string>> correct = {{
          {0, "zero"},
          {1, "one"},
          {2, "two"},
      }};
      CHECK(result == correct);
    }

    SECTION("mismatched key") {
      d1.insert({2, "three"});
      std::optional<std::unordered_map<int, std::string>> result =
          try_merge_nondisjoint_unordered_maps(d1, d2);
      std::optional<std::unordered_map<int, std::string>> correct =
          std::nullopt;
      CHECK(result == correct);
    }

    SECTION("repeated value") {
      d1.insert({3, "one"});
      std::optional<std::unordered_map<int, std::string>> result =
          try_merge_nondisjoint_unordered_maps(d1, d2);
      std::optional<std::unordered_map<int, std::string>> correct = {{
          {0, "zero"},
          {1, "one"},
          {2, "two"},
          {3, "one"},
      }};
      CHECK(result == correct);
    }

    SECTION("left superset") {
      d1.insert({2, "two"});
      std::optional<std::unordered_map<int, std::string>> result =
          try_merge_nondisjoint_unordered_maps(d1, d2);
      std::optional<std::unordered_map<int, std::string>> correct = d1;
      CHECK(result == correct);
    }

    SECTION("right superset") {
      d2.insert({1, "one"});
      std::optional<std::unordered_map<int, std::string>> result =
          try_merge_nondisjoint_unordered_maps(d1, d2);
      std::optional<std::unordered_map<int, std::string>> correct = d2;
      CHECK(result == correct);
    }

    SECTION("equal") {
      d1.insert({2, "two"});
      d2.insert({1, "one"});
      std::optional<std::unordered_map<int, std::string>> result =
          try_merge_nondisjoint_unordered_maps(d1, d2);
      std::optional<std::unordered_map<int, std::string>> correct = d1;
      CHECK(result == correct);
    }
  }
