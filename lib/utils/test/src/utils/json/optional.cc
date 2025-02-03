#include "utils/json/optional.h"
#include "test/utils/doctest/fmt/optional.h"
#include <catch2/catch_test_macros.hpp>

using namespace ::FlexFlow;


  TEST_CASE("adl_serializer<std::optional<T>>") {
    SECTION("to_json") {
      SECTION("has value") {
        std::optional<int> input = 5;

        nlohmann::json result = input;
        nlohmann::json correct = 5;

        CHECK(result == correct);
      }

      SECTION("has nullopt") {
        std::optional<int> input = std::nullopt;

        nlohmann::json result = input;
        nlohmann::json correct = nullptr;

        CHECK(result == correct);
      }
    }

    SECTION("from_json") {
      SECTION("has value") {
        nlohmann::json input = 5;

        std::optional<int> result = input;
        std::optional<int> correct = 5;

        CHECK(result == correct);
      }

      SECTION("has nullopt") {
        nlohmann::json input = nullptr;

        std::optional<int> result = input.get<std::optional<int>>();
        std::optional<int> correct = std::nullopt;

        CHECK(result == correct);
      }
    }
  }
