#include "utils/record_formatter.h"
#include <catch2/catch_test_macros.hpp>

std::string formatRecord(RecordFormatter const &formatter) {
  std::ostringstream oss;
  oss << formatter;
  return oss.str();
}


  TEST_CASE("RecordFormatter") {
    RecordFormatter formatter;
    SECTION("Appending string") {
      formatter << "Hello";
      formatter << "World";
      CHECK(formatRecord(formatter) == "{ Hello | World }");
    }

    SECTION("Appending integer and float") {
      formatter << 42;
      formatter << 3.14f;
      CHECK(formatRecord(formatter) == "{ 42 | 3.140000e+00 }");
    }

    SECTION("Appending another RecordFormatter") {
      RecordFormatter subFormatter;
      subFormatter << "Sub";
      subFormatter << "Formatter";

      RecordFormatter formatter;
      formatter << "Hello";
      formatter << subFormatter;

      std::ostringstream oss;
      oss << formatter;

      CHECK(formatRecord(formatter) == "{ Hello | { Sub | Formatter } }");
    }
  }
