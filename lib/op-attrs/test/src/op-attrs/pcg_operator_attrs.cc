#include "op-attrs/pcg_operator_attrs.dtg.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;


  TEST_CASE("PCGOperatorAttrs to/from json") {
    PCGOperatorAttrs correct = PCGOperatorAttrs{RepartitionAttrs{
        /*repartition_dim=*/ff_dim_t{1_n},
        /*repartition_degree=*/4_n,
    }};
    nlohmann::json j = correct;
    auto result = j.get<PCGOperatorAttrs>();

    CHECK(result == correct);
  }
}
