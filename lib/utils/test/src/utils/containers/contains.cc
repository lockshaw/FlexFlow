#include "utils/containers/contains.h"
#include <catch2/catch_test_macros.hpp>
#include <vector>

using namespace FlexFlow;


  TEST_CASE("contains") {
    std::vector<int> v = {1, 2, 3, 4, 5};
    CHECK(contains(v, 3));
    CHECK(!contains(v, 6));
  }
