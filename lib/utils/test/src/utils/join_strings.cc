#include "utils/join_strings.h"
#include <catch2/catch_test_macros.hpp>
#include <string>
#include <vector>

using namespace FlexFlow;



  TEST_CASE("join_strings") {
    std::vector<std::string> v = {"Hello", "world", "!"};

    SECTION("iterator") {
      std::string result = join_strings(v.begin(), v.end(), " ");
      std::string correct = "Hello world !";
      CHECK(result == correct);
    }

    SECTION("join_strings with container") {
      std::string result = join_strings(v, " ");
      std::string correct = "Hello world !";
      CHECK(result == correct);
    }

    SECTION("join_strings with transforming function") {
      auto add_exclamation = [](std::string const &str) { return str + "!"; };
      std::string result = join_strings(v, " ", add_exclamation);
      std::string correct = "Hello! world! !!";
      CHECK(result == correct);
    }

    SECTION("join_strings with transforming function, iterator") {
      auto add_exclamation = [](std::string const &str) { return str + "!"; };
      std::string result =
          join_strings(v.begin(), v.end(), " ", add_exclamation);
      std::string correct = "Hello! world! !!";
      CHECK(result == correct);
    }

    SECTION("empty sequence") {
      v = {};
      std::string result = join_strings(v, "!");
      std::string correct = "";
      CHECK(result == correct);
    }
  }
