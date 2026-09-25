#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_BINARY_CARTESIAN_PRODUCT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_BINARY_CARTESIAN_PRODUCT_H

#include <set>
#include <libassert/assert.hpp>

namespace FlexFlow {

template <typename T>
inline constexpr bool is_forward_iterator_v = 
  std::is_base_of_v<
    std::forward_iterator_tag,
    typename std::iterator_traits<T>::iterator_category
  >;

template <typename It1, typename It2>
struct binary_cartesian_product_container {
  static_assert(is_forward_iterator_v<It1>);
  static_assert(is_forward_iterator_v<It2>);

public:
  binary_cartesian_product_container(
    It1 start1, It1 end1,
    It2 start2, It2 end2)
  : start1(start1),
    end1(end1),
    start2(start2),
    end2(end2)
  { }

  struct iterator {
  private:
    using A = typename It1::value_type;
    using B = typename It2::value_type;

  public:
    using difference_type = long;
    using value_type = std::pair<A, B>;
    using pointer = std::pair<A, B> const *;
    using reference = std::pair<A, B> const &;
    using iterator_category = std::input_iterator_tag;

  public:
    explicit iterator(
        binary_cartesian_product_container<It1, It2> const &c,
        It1 it1,
        It2 it2)
      : c(c), it1(it1), it2(it2), curr(std::nullopt)
    {}

    iterator &operator++() {
      ASSERT(it1 != this->c.end1 || it2 != this->c.end2);

      it2++;
      if (it2 == this->c.end2) {
        it1++;

        if (it1 != this->c.end1) {
          it2 = this->c.start2;
        }
      }

      return *this;
    }

    iterator operator++(int) {
      iterator retval = *this;
      ++(*this);
      return retval;
    }

    bool operator==(iterator other) const {
      return &this->c == &other.c && this->it1 == other.it1 && this->it2 == other.it2;
    }

    bool operator!=(iterator other) const {
      return &this->c != &other.c || this->it1 != other.it1 || this->it2 != other.it2;
    }

    reference operator*() const {
      ASSERT(it1 != this->c.end1 || it2 != this->c.end2);

      if (!this->curr.has_value()) {
        this->curr = std::pair<A, B>{*it1, *it2};
      } else {
        this->curr->first = *it1;
        this->curr->second = *it2;
      }

      return *(this->curr);
    }
  private:
    It1 it1;
    It2 it2;
    mutable std::optional<std::pair<A, B>> curr;

    binary_cartesian_product_container<It1, It2> const &c;
  };

  using const_iterator = iterator;
  using value_type = typename iterator::value_type;
  using difference_type = typename iterator::difference_type;
  using pointer = typename iterator::pointer;
  using reference = typename iterator::reference;
  using const_reference = typename iterator::reference;

  iterator begin() const {
    if (start1 == end1 || start2 == end2) {
      return iterator(*this, this->end1, this->end2);
    } else {
      return iterator(*this, this->start1, this->start2);
    }
  }

  iterator end() const {
    return iterator(*this, this->end1, this->end2);
  }

  const_iterator cbegin() const {
    return this->begin();
  }

  const_iterator cend() const {
    return this->end();
  }

private:
  It1 start1;
  It1 end1;

  It2 start2;
  It2 end2;
};

template <typename A, typename B>
binary_cartesian_product_container<
  typename std::set<A>::const_iterator,
  typename std::set<B>::const_iterator
>
  binary_cartesian_product(std::set<A> const &lhs,
                           std::set<B> const &rhs)
{
  return binary_cartesian_product_container(lhs.cbegin(), lhs.cend(), rhs.cbegin(), rhs.cend());
}

} // namespace FlexFlow

#endif
