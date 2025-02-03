#include "utils/nonnegative_int/num_elements.h"
#include <catch2/catch_test_macros.hpp>

using namespace ::FlexFlow;


  TEST_CASE("num_elements") {
    std::vector<int> input = {-1, 3, 3, 1};

    nonnegative_int result = num_elements(input);
    nonnegative_int correct = nonnegative_int{4};

    CHECK(result == correct);
  }
