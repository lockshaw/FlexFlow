#include "utils/fmt/variant.h"
#include <catch2/catch_test_macros.hpp>

using namespace ::FlexFlow;


  TEST_CASE("fmt::to_string(std::variant<int, std::string>)") {
    SECTION("has int") {
      std::variant<int, std::string> input = 4;
      std::string result = fmt::to_string(input);
      std::string correct = "4";
      CHECK(result == correct);
    }

    SECTION("has string") {
      std::variant<int, std::string> input = "hello world";
      std::string result = fmt::to_string(input);
      std::string correct = "hello world";
      CHECK(result == correct);
    }
  }
