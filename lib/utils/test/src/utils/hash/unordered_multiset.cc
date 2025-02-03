#include "utils/hash/unordered_multiset.h"
#include <catch2/catch_test_macros.hpp>

using namespace ::FlexFlow;


  TEST_CASE("std::hash<std::unordered_multiset<int>>") {
    std::unordered_multiset<int> input = {1, 2, 2, 1, 5};
    size_t input_hash = get_std_hash(input);

    SECTION("same values have the same hash") {
      std::unordered_multiset<int> also_input = {2, 1, 2, 5, 1};
      size_t also_input_hash = get_std_hash(input);

      CHECK(input_hash == also_input_hash);
    }

    SECTION("different values have different hashes") {
      SECTION("different number of duplicates") {
        std::unordered_multiset<int> other = {1, 2, 2, 1, 5, 5};
        size_t other_hash = get_std_hash(other);

        CHECK(input_hash != other_hash);
      }

      SECTION("different elements") {
        std::unordered_multiset<int> other = {1, 2, 2, 1, 6};
        size_t other_hash = get_std_hash(other);

        CHECK(input_hash != other_hash);
      }
    }
  }
