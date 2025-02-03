#include "utils/containers/recurse_n.h"
#include <catch2/catch_test_macros.hpp>
#include <string>

using namespace FlexFlow;


  TEST_CASE("recurse_n") {
    auto append_bar = [](std::string const &x) {
      return x + std::string("Bar");
    };

    SECTION("n = 0") {
      std::string result = recurse_n(append_bar, 0, std::string("Foo"));
      std::string correct = "Foo";
      CHECK(result == correct);
    }

    SECTION("n = 3") {
      std::string result = recurse_n(append_bar, 3, std::string("Foo"));
      std::string correct = "FooBarBarBar";
      CHECK(result == correct);
    }

    SECTION("n < 0") {
      CHECK_THROWS(recurse_n(append_bar, -1, std::string("Foo")));
    }
  }
