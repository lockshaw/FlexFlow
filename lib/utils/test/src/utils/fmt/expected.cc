#include "utils/fmt/expected.h"
#include "test/utils/doctest/fmt/expected.h"
#include <catch2/catch_test_macros.hpp>

using namespace ::FlexFlow;


  TEST_CASE("fmt::to_string(tl::expected<int, std::string>)") {
    SECTION("expected") {
      tl::expected<int, std::string> input = 4;
      std::string result = fmt::to_string(input);
      std::string correct = "expected(4)";
      CHECK(result == correct);
    }

    SECTION("unexpected") {
      tl::expected<int, std::string> input = tl::unexpected("hello world");
      std::string result = fmt::to_string(input);
      std::string correct = "unexpected(hello world)";
      CHECK(result == correct);
    }
  }
