#include "utils/type_index.h"
#include "test/utils/doctest/check_without_stringify.h"
#include <catch2/catch_test_macros.hpp>
#include <typeindex>

using namespace ::FlexFlow;


  TEST_CASE("get_type_index_for_type") {
    SECTION("int type") {
      std::type_index idx = get_type_index_for_type<int>();
      std::type_index expected_idx = typeid(int);
      CHECK_WITHOUT_STRINGIFY(idx == expected_idx);
    }

    SECTION("string type") {
      std::type_index idx = get_type_index_for_type<std::string>();
      std::type_index expected_idx = typeid(std::string);
      CHECK_WITHOUT_STRINGIFY(idx == expected_idx);
    }
  }

  TEST_CASE("matches<T>(std::type_index)") {
    std::type_index idx = typeid(float);

    SECTION("matching type") {
      CHECK(matches<float>(idx));
    }

    SECTION("non-matching type") {
      CHECK_FALSE(matches<int>(idx));
    }
  }
