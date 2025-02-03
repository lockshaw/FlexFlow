#include "utils/rapidcheck/optional.h"
#include "test/utils/rapidcheck.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <rapidcheck.h>
#include <rapidcheck/catch.h>

using namespace ::FlexFlow;


  TEMPLATE_TEST_CASE(
      "Arbitrary<std::optional<TestType>> with TestType=", "", int, double, char) {
    rc::prop("generation works", [](std::optional<TestType> o) { });
  }
