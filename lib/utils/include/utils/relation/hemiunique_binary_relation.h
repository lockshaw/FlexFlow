#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_RELATION_HEMIUNIQUE_RELATION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_RELATION_HEMIUNIQUE_RELATION_H

#include "utils/bidict/algorithms/bidict_from_unstructured_relation.h"
#include "utils/bidict/algorithms/unstructured_relation_from_bidict.h"
#include "utils/fmt/variant.h"
#include "utils/many_to_one/invert_many_to_one.h"
#include "utils/many_to_one/many_to_one_is_biunique.h"
#include "utils/one_to_many/invert_one_to_many.h"
#include "utils/one_to_many/one_to_many_is_biunique.h"
#include "utils/overload.h"
#include "utils/relation/uniqueness.dtg.h"

namespace FlexFlow {

template <typename L, typename R>
struct HemiuniqueBinaryRelation {
  HemiuniqueBinaryRelation() : raw(bidict<L, R>{}) {}

  explicit HemiuniqueBinaryRelation(bidict<L, R> const &b) : raw(b) {}

  explicit HemiuniqueBinaryRelation(OneToMany<L, R> const &otm) {
    if (one_to_many_is_biunique(otm)) {
      this->raw = bidict_from_unstructured_relation(
          unstructured_relation_from_one_to_many(otm));
    } else {
      this->raw = otm;
    }
  }

  explicit HemiuniqueBinaryRelation(ManyToOne<L, R> const &mto) {
    if (many_to_one_is_biunique(mto)) {
      this->raw = bidict_from_unstructured_relation(
          unstructured_relation_from_many_to_one(mto));
    } else {
      this->raw = mto;
    }
  }

  bool operator==(HemiuniqueBinaryRelation const &other) const {
    return this->raw == other.raw;
  }

  bool operator!=(HemiuniqueBinaryRelation const &other) const {
    return this->raw != other.raw;
  }

  bool operator<(HemiuniqueBinaryRelation const &other) const;
  bool operator>(HemiuniqueBinaryRelation const &other) const;
  bool operator<=(HemiuniqueBinaryRelation const &other) const;
  bool operator>=(HemiuniqueBinaryRelation const &other) const;

  template <typename ReturnType, typename Visitor>
  ReturnType visit(Visitor &&v) const {
    switch (this->raw.index()) {
      case 0: {
        ReturnType result = v(std::get<bidict<L, R>>(this->raw));
        return result;
      }
      case 1: {
        ReturnType result = v(std::get<OneToMany<L, R>>(this->raw));
        return result;
      }
      case 2: {
        ReturnType result = v(std::get<ManyToOne<L, R>>(this->raw));
        return result;
      }
      default: {
        PANIC("Unknown index {} for type HemiuniqueBinaryRelation",
              this->raw.index());
      }
    };
  }

  std::set<std::pair<L, R>> as_unstructured_relation() const {
    return this->visit<std::set<std::pair<L, R>>>(overload{
        [](bidict<L, R> const &b) -> std::set<std::pair<L, R>> {
          return unstructured_relation_from_bidict(b);
        },
        [](OneToMany<L, R> const &otm) -> std::set<std::pair<L, R>> {
          return unstructured_relation_from_one_to_many(otm);
        },
        [](ManyToOne<L, R> const &mto) -> std::set<std::pair<L, R>> {
          return unstructured_relation_from_many_to_one(mto);
        },
    });
  }

  Uniqueness get_uniqueness() const {
    return this->visit<Uniqueness>(overload{
        [](bidict<L, R> const &) -> Uniqueness { return Uniqueness::BIUNIQUE; },
        [](OneToMany<L, R> const &) -> Uniqueness {
          return Uniqueness::LEFT_UNIQUE;
        },
        [](ManyToOne<L, R> const &) -> Uniqueness {
          return Uniqueness::RIGHT_UNIQUE;
        },
    });
  }

  std::set<L> left_entries() const {
    return this->visit<std::set<L>>(overload{
        [](bidict<L, R> const &b) -> std::set<L> { return b.left_values(); },
        [](OneToMany<L, R> const &otm) -> std::set<L> {
          return otm.left_values();
        },
        [](ManyToOne<L, R> const &mto) -> std::set<L> {
          return mto.left_values();
        },
    });
  }

  std::set<R> right_entries() const {
    return this->visit<std::set<R>>(overload{
        [](bidict<L, R> const &b) -> std::set<R> { return b.right_values(); },
        [](OneToMany<L, R> const &otm) -> std::set<R> {
          return otm.right_values();
        },
        [](ManyToOne<L, R> const &mto) -> std::set<R> {
          return mto.right_values();
        },
    });
  }

  HemiuniqueBinaryRelation<R, L> inverted() const {
    return this->visit<HemiuniqueBinaryRelation<R, L>>(overload{
        [](bidict<L, R> const &b) -> HemiuniqueBinaryRelation<R, L> {
          return HemiuniqueBinaryRelation<R, L>{
              b.reversed(),
          };
        },
        [](OneToMany<L, R> const &otm) -> HemiuniqueBinaryRelation<R, L> {
          return HemiuniqueBinaryRelation<R, L>{
              invert_one_to_many(otm),
          };
        },
        [](ManyToOne<L, R> const &mto) -> HemiuniqueBinaryRelation<R, L> {
          return HemiuniqueBinaryRelation<R, L>{
              invert_many_to_one(mto),
          };
        },
    });
  }

  bidict<L, R> const &require_biunique() const {
    ASSERT(this->get_uniqueness() == Uniqueness::BIUNIQUE);

    return std::get<bidict<L, R>>(this->raw);
  }

  OneToMany<L, R> const &require_strictly_left_unique() const {
    ASSERT(this->get_uniqueness() == Uniqueness::LEFT_UNIQUE);

    return std::get<OneToMany<L, R>>(this->raw);
  }

  ManyToOne<L, R> const &require_strictly_right_unique() const {
    ASSERT(this->get_uniqueness() == Uniqueness::RIGHT_UNIQUE);

    return std::get<ManyToOne<L, R>>(this->raw);
  }

  template <typename LL, typename RR>
  friend std::string format_as(HemiuniqueBinaryRelation<LL, RR> const &);

  friend struct std::hash<HemiuniqueBinaryRelation>;

private:
  std::variant<bidict<L, R>, OneToMany<L, R>, ManyToOne<L, R>> raw;
};

template <typename L, typename R>
std::string format_as(HemiuniqueBinaryRelation<L, R> const &x) {
  return fmt::format("<HemiuniqueBinaryRelation raw={}>", x.raw);
}

template <typename L, typename R>
std::ostream &operator<<(std::ostream &s,
                         HemiuniqueBinaryRelation<L, R> const &x) {
  return (s << fmt::to_string(x));
}

} // namespace FlexFlow

namespace std {

template <typename L, typename R>
struct hash<::FlexFlow::HemiuniqueBinaryRelation<L, R>> {
  size_t operator()(
      ::FlexFlow::HemiuniqueBinaryRelation<L, R> const &r) const noexcept {
    return ::FlexFlow::get_std_hash(r.raw);
  }
};

} // namespace std

#endif
