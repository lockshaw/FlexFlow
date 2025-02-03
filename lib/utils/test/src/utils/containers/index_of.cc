#include "utils/containers/index_of.h"
#include "test/utils/doctest/fmt/optional.h"
#include <catch2/catch_test_macros.hpp>
#include <optional>
#include <vector>

using namespace ::FlexFlow;


  TEST_CASE("index_of") {

    std::vector<int> v = {1, 2, 3, 4, 3, 5};

    SECTION("element occurs once in container") {
      CHECK(index_of(v, 4).value() == 3);
    }
    SECTION("if element appears multiple times, return the first occurrence") {
      CHECK(index_of(v, 3).value() == 2);
    }
    SECTION("element not in container") {
      CHECK(index_of(v, 7) == std::nullopt);
    }
  }
