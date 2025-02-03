#include "utils/rapidcheck/variant.h"
#include "test/utils/rapidcheck.h"
#include <catch2/catch_test_macros.hpp>
#include <rapidcheck.h>
#include <rapidcheck/catch.h>

using namespace ::FlexFlow;


  TEST_CASE("Arbitrary<std::variant>") {
    rc::prop("valid type", [](std::variant<int, float> v) {
      return std::holds_alternative<int>(v) || std::holds_alternative<float>(v);
    });
  }
