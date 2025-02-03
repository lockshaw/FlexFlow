#include "utils/containers/to_uppercase.h"
#include <catch2/catch_test_macros.hpp>

using namespace ::FlexFlow;


  TEST_CASE("to_uppercase(std::string)") {
    std::string input = "Hello World";

    std::string result = to_uppercase(input);
    std::string correct = "HELLO WORLD";

    CHECK(result == correct);
  }
