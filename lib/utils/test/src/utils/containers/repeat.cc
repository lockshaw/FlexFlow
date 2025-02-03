#include "utils/containers/repeat.h"
#include "test/utils/doctest/fmt/vector.h"
#include <catch2/catch_test_macros.hpp>

using namespace ::FlexFlow;


  TEST_CASE("repeat") {
    int x = 0;
    std::vector<int> result = repeat(3_n, [&]() {
      int result = x;
      x += 2;
      return result;
    });

    std::vector<int> correct = {0, 2, 4};

    CHECK(result == correct);
  }
