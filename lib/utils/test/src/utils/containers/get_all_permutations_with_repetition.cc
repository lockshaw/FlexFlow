#include "utils/containers/get_all_permutations_with_repetition.h"
#include "test/utils/doctest/fmt/unordered_multiset.h"
#include "test/utils/doctest/fmt/vector.h"
#include "utils/hash/vector.h"
#include <catch2/catch_test_macros.hpp>

using namespace ::FlexFlow;



  TEST_CASE("get_all_permutations_with_repetition") {
    SECTION("output vector has only one element") {
      std::vector<int> input = {1, 2, 3};

      std::unordered_multiset<std::vector<int>> result =
          get_all_permutations_with_repetition(input, 1_n);
      std::unordered_multiset<std::vector<int>> correct = {
          {1},
          {2},
          {3},
      };

      CHECK(result == correct);
    }

    SECTION("input vector has only one element") {
      std::vector<int> input = {1};

      std::unordered_multiset<std::vector<int>> result =
          get_all_permutations_with_repetition(input, 2_n);
      std::unordered_multiset<std::vector<int>> correct = {
          {1, 1},
      };

      CHECK(result == correct);
    }

    SECTION("input, output vectors have more than 1 element") {
      std::vector<int> input = {1, 2};

      std::unordered_multiset<std::vector<int>> result =
          get_all_permutations_with_repetition(input, 3_n);
      std::unordered_multiset<std::vector<int>> correct = {
          {1, 1, 1},
          {1, 1, 2},
          {1, 2, 1},
          {1, 2, 2},
          {2, 1, 1},
          {2, 1, 2},
          {2, 2, 1},
          {2, 2, 2},
      };

      CHECK(result == correct);
    }

    SECTION("duplicate elements") {
      std::vector<int> input = {1, 2, 2};

      std::unordered_multiset<std::vector<int>> result =
          get_all_permutations_with_repetition(input, 2_n);
      std::unordered_multiset<std::vector<int>> correct = {{1, 1},
                                                           {1, 2},
                                                           {1, 2},
                                                           {2, 1},
                                                           {2, 1},
                                                           {2, 2},
                                                           {2, 2},
                                                           {2, 2},
                                                           {2, 2}};

      CHECK(result == correct);
    }
  }
