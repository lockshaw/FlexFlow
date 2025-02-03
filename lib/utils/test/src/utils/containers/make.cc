#include "utils/containers/make.h"
#include <catch2/catch_test_macros.hpp>

using namespace ::FlexFlow;


  TEST_CASE("make") {
    auto f = make<int>();

    int result = f(true);
    int correct = 1;

    CHECK(result == correct);
  }
