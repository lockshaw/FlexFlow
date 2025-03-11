#include "op-attrs/datatype_value.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("make_float_data_type_value") {
    double value = 1.0f;
    DataTypeValue data_type_value = make_float_data_type_value(value);

    CHECK(data_type_value.has<float>());
    CHECK(data_type_value.get<float>() == value);
  }

  TEST_CASE("make_double_data_type_value") {
    double value = 2.71828;
    DataTypeValue data_type_value = make_double_data_type_value(value);

    CHECK(data_type_value.has<double>());
    CHECK(data_type_value.get<double>() == value);
  }

  TEST_CASE("make_int32_data_type_value") {
    int32_t value = -42;
    DataTypeValue data_type_value = make_int32_data_type_value(value);

    CHECK(data_type_value.has<int32_t>());
    CHECK(data_type_value.get<int32_t>() == value);
  }

  TEST_CASE("make_int64_data_type_value") {
    int64_t value = 1LL << 40;
    DataTypeValue data_type_value = make_int64_data_type_value(value);

    CHECK(data_type_value.has<int64_t>());
    CHECK(data_type_value.get<int64_t>() == value);
  }

  TEST_CASE("make_bool_data_type_value") {
    bool value = true;
    DataTypeValue data_type_value = make_bool_data_type_value(value);

    CHECK(data_type_value.has<bool>());
    CHECK(data_type_value.get<bool>() == value);
  }
}
