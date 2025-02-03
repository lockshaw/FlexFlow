#include "utils/containers/require_no_duplicates.h"
#include "test/utils/doctest/fmt/multiset.h"
#include "test/utils/doctest/fmt/set.h"
#include "test/utils/doctest/fmt/unordered_multiset.h"
#include "test/utils/doctest/fmt/unordered_set.h"
#include <catch2/catch_test_macros.hpp>

using namespace ::FlexFlow;


  TEST_CASE("require_no_duplicates(std::unordered_multiset<T>)") {
    SECTION("empty") {
      std::unordered_multiset<int> input = {};

      std::unordered_set<int> result = require_no_duplicates(input);
      std::unordered_set<int> correct = {};

      CHECK(result == correct);
    }

    SECTION("input has duplicates") {
      std::unordered_multiset<int> input = {1, 2, 2};

      CHECK_THROWS(require_no_duplicates(input));
    }

    SECTION("input does not have duplicates") {
      std::unordered_multiset<int> input = {1, 2, 4};

      std::unordered_set<int> result = require_no_duplicates(input);
      std::unordered_set<int> correct = {1, 2, 4};

      CHECK(result == correct);
    }
  }

  TEST_CASE("require_no_duplicates(std::multiset<T>)") {
    SECTION("empty") {
      std::multiset<int> input = {};

      std::set<int> result = require_no_duplicates(input);
      std::set<int> correct = {};

      CHECK(result == correct);
    }

    SECTION("input has duplicates") {
      std::multiset<int> input = {1, 2, 2};

      CHECK_THROWS(require_no_duplicates(input));
    }

    SECTION("input does not have duplicates") {
      std::multiset<int> input = {1, 2, 4};

      std::set<int> result = require_no_duplicates(input);
      std::set<int> correct = {1, 2, 4};

      CHECK(result == correct);
    }
  }
