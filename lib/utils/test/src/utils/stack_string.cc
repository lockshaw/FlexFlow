#include "utils/stack_string.h"
#include "test/utils/rapidcheck.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <rapidcheck.h>
#include <rapidcheck/catch.h>

using namespace FlexFlow;


  TEMPLATE_TEST_CASE("StackStringConstruction", "", char) {
    constexpr std::size_t MAXSIZE = 5;
    using StackString = stack_string<MAXSIZE>;

    SECTION("DefaultConstruction") {
      StackString str;
      CHECK(str.size() == 0);
      CHECK(str.length() == 0);
      CHECK(static_cast<std::string>(str) == "");
    }

    SECTION("CStringConstruction") {
      char const *cstr = "Hello";
      StackString str(cstr);
      CHECK(str.size() == 5);
      CHECK(str.length() == 5);
      CHECK(static_cast<std::string>(str) == "Hello");
    }

    SECTION("ShortCStringConstruction") {
      char const *cstr = "CMU";
      StackString str(cstr);
      CHECK(str.size() == 3);
      CHECK(str.length() == 3);
      CHECK(static_cast<std::string>(str) == "CMU");
    }

    SECTION("StdStringConstruction") {
      std::basic_string<TestType> stdStr = "World";
      StackString str(stdStr);
      CHECK(str.size() == 5);
      CHECK(str.length() == 5);
      CHECK(static_cast<std::string>(str) == "World");
    }
  }

  TEST_CASE("StackStringComparison") {
    constexpr std::size_t MAXSIZE = 5;
    using StackString = stack_string<MAXSIZE>;

    StackString str1{"abc"};
    StackString str2{"def"};
    StackString str3{"abc"};

    CHECK(str1 == str1);
    CHECK(str1 == str3);
    CHECK(str1 != str2);
    CHECK(str2 != str3);
    CHECK(str1 < str2);
  }

  TEST_CASE("StackStringSize") {
    constexpr std::size_t MAXSIZE = 5;
    using StackString = stack_string<MAXSIZE>;

    SECTION("EmptyString") {
      StackString str;
      CHECK(str.size() == 0);
      CHECK(str.length() == 0);
    }

    SECTION("NonEmptyString") {
      StackString str{"Hello"};
      CHECK(str.size() == 5);
      CHECK(str.length() == 5);
    }
  }

  TEST_CASE("StackStringConversion") {
    constexpr std::size_t MAXSIZE = 5;
    using StackString = stack_string<MAXSIZE>;

    StackString str{"Hello"};
    std::string stdStr = static_cast<std::string>(str);
    CHECK(stdStr == "Hello");
  }

  TEST_CASE("Arbitrary<stack_string>") {
    constexpr std::size_t MAXSIZE = 10;
    rc::prop("generated values are under MAXSIZE", [&](stack_string<MAXSIZE> const &s) {
      return s.size() <= MAXSIZE;
    });
  }
