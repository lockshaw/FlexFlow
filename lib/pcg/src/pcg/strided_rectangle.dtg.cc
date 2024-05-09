// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/pcg/include/pcg/strided_rectangle.struct.toml
/* proj-data
{
  "generated_from": "87af84e6a16d5363049cb9a9a75e4f5f"
}
*/

#include "pcg/strided_rectangle.dtg.h"

#include "op-attrs/dim_ordered.h"
#include "pcg/strided_rectangle_side.dtg.h"
#include <sstream>

namespace FlexFlow {
StridedRectangle::StridedRectangle(
    ::FlexFlow::FFOrdered<::FlexFlow::StridedRectangleSide> const &sides)
    : sides(sides) {}
bool StridedRectangle::operator==(StridedRectangle const &other) const {
  return std::tie(this->sides) == std::tie(other.sides);
}
bool StridedRectangle::operator!=(StridedRectangle const &other) const {
  return std::tie(this->sides) != std::tie(other.sides);
}
bool StridedRectangle::operator<(StridedRectangle const &other) const {
  return std::tie(this->sides) < std::tie(other.sides);
}
bool StridedRectangle::operator>(StridedRectangle const &other) const {
  return std::tie(this->sides) > std::tie(other.sides);
}
bool StridedRectangle::operator<=(StridedRectangle const &other) const {
  return std::tie(this->sides) <= std::tie(other.sides);
}
bool StridedRectangle::operator>=(StridedRectangle const &other) const {
  return std::tie(this->sides) >= std::tie(other.sides);
}
} // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::StridedRectangle>::operator()(
    FlexFlow::StridedRectangle const &x) const {
  size_t result = 0;
  result ^=
      std::hash<::FlexFlow::FFOrdered<::FlexFlow::StridedRectangleSide>>{}(
          x.sides) +
      0x9e3779b9 + (result << 6) + (result >> 2);
  return result;
}
} // namespace std

namespace nlohmann {
FlexFlow::StridedRectangle
    adl_serializer<FlexFlow::StridedRectangle>::from_json(json const &j) {
  return {j.at("sides")
              .template get<
                  ::FlexFlow::FFOrdered<::FlexFlow::StridedRectangleSide>>()};
}
void adl_serializer<FlexFlow::StridedRectangle>::to_json(
    json &j, FlexFlow::StridedRectangle const &v) {
  j["__type"] = "StridedRectangle";
  j["sides"] = v.sides;
}
} // namespace nlohmann

namespace FlexFlow {
std::string format_as(StridedRectangle const &x) {
  std::ostringstream oss;
  oss << "<StridedRectangle";
  oss << " sides=" << x.sides;
  oss << ">";
  return oss.str();
}
std::ostream &operator<<(std::ostream &s, StridedRectangle const &x) {
  return s << fmt::to_string(x);
}
} // namespace FlexFlow
