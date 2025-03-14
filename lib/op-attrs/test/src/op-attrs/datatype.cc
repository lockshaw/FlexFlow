#include "op-attrs/datatype.h"
#include "test/utils/rapidcheck.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("can_strictly_promote_datatype_from_to(DataType, DataType)") {
    CHECK(
        can_strictly_promote_datatype_from_to(DataType::BOOL, DataType::INT32));
    CHECK(can_strictly_promote_datatype_from_to(DataType::INT32,
                                                DataType::INT64));
    CHECK(can_strictly_promote_datatype_from_to(DataType::FLOAT,
                                                DataType::DOUBLE));

    RC_SUBCASE("is strict", [](DataType d) {
      RC_ASSERT(!can_strictly_promote_datatype_from_to(d, d));
    });

    RC_SUBCASE("is asymmetric", [](DataType l, DataType r) {
      RC_PRE(can_strictly_promote_datatype_from_to(l, r));
      RC_ASSERT(!can_strictly_promote_datatype_from_to(r, l));
    });

    RC_SUBCASE("is transitive", [](DataType d1, DataType d2, DataType d3) {
      RC_PRE(can_strictly_promote_datatype_from_to(d1, d2));
      RC_PRE(can_strictly_promote_datatype_from_to(d2, d3));
      RC_ASSERT(can_strictly_promote_datatype_from_to(d1, d3));
    });

    RC_SUBCASE("is stronger than torch casting", [](DataType d1, DataType d2) {
      RC_PRE(can_strictly_promote_datatype_from_to(d1, d2));
      RC_ASSERT(can_torch_strictly_promote_datatype_from_to(d1, d2));
    });
  }

  TEST_CASE("can_torch_strictly_promote_datatype_from_to(DataType, DataType)") {
    CHECK(can_torch_strictly_promote_datatype_from_to(DataType::BOOL,
                                                      DataType::INT32));
    CHECK(can_torch_strictly_promote_datatype_from_to(DataType::INT32,
                                                      DataType::INT64));
    CHECK(can_torch_strictly_promote_datatype_from_to(DataType::FLOAT,
                                                      DataType::DOUBLE));

    RC_SUBCASE("is strict", [](DataType d) {
      RC_ASSERT(!can_torch_strictly_promote_datatype_from_to(d, d));
    });

    RC_SUBCASE("is transitive if end-points are not the same",
               [](DataType d1, DataType d2, DataType d3) {
                 RC_PRE(can_torch_strictly_promote_datatype_from_to(d1, d2));
                 RC_PRE(can_torch_strictly_promote_datatype_from_to(d2, d3));
                 RC_PRE(d1 != d3);
                 RC_ASSERT(can_torch_strictly_promote_datatype_from_to(d1, d3));
               });
  }
}
