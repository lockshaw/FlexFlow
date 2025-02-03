#include "utils/containers/all_of.h"
#include <fmt/format.h>
#include <catch2/catch_test_macros.hpp>
#include <vector>

using namespace FlexFlow;


  TEST_CASE("all_of") {
    std::vector<int> v = {2, 4, 6, 8};

    SECTION("result is true") {
      bool result = all_of(v, [](int x) { return x % 2 == 0; });
      bool correct = true;
      CHECK(result == correct);
    }

    SECTION("result is false") {
      bool result = all_of(v, [](int x) { return x % 4 == 0; });
      bool correct = false;
      CHECK(result == correct);
    }
  }
