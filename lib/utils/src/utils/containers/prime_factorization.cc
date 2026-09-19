#include "utils/containers/prime_factorization.h"

namespace FlexFlow {

std::multiset<int_ge_two>
  prime_factorization(positive_int x) 
{
  if (x == 1) {
    return {};
  } else {
    return prime_factorization(int_ge_two{x});
  }
}

std::multiset<int_ge_two>
  prime_factorization(int_ge_two x)
{
  std::multiset<int_ge_two> factors;
  positive_int remaining = x.positive_int_from_int_ge_two();

  for (int_ge_two candidate : int_ge_two_range(x)) {
    while (remaining % candidate == 0) {
      remaining /= candidate;
      factors.insert(candidate)
    }
  }

  return factors;
}

} // namespace FlexFlow
