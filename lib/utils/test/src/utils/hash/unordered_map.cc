#include "utils/hash/unordered_map.h"
#include <catch2/catch_test_macros.hpp>

using namespace ::FlexFlow;


  TEST_CASE("std::hash<std::unordered_map<K, V>>") {
    std::unordered_map<int, int> map1{{1, 2}};
    std::unordered_map<int, int> map2{{1, 2}, {3, 4}};

    size_t hash1 = get_std_hash(map1);
    size_t hash2 = get_std_hash(map2);

    CHECK(hash1 != hash2);

    map1.insert({3, 4});
    hash1 = get_std_hash(map1);
    CHECK(hash1 == hash2);
  }
