#include "utils/graph/multidigraph/algorithms/get_outgoing_edges.h"
#include "utils/containers/group_by.h"
#include "utils/graph/multidigraph/algorithms/get_edges.h"
#include "utils/graph/node/algorithms.h"
#include <unordered_set>
#include "utils/containers/set_of.h"
namespace FlexFlow {

std::unordered_set<MultiDiEdge> get_outgoing_edges(MultiDiGraphView const &g,
                                                   Node const &n) {
  MultiDiEdgeQuery query = MultiDiEdgeQuery{
    query_set<Node>::match_single_value(n),
    query_set<Node>::matchall(),
  };

  return g.query_edges(query);
}

std::unordered_map<Node, std::unordered_set<MultiDiEdge>>
    get_outgoing_edges(MultiDiGraphView const &g,
                       std::unordered_set<Node> const &ns) {
  MultiDiEdgeQuery query = MultiDiEdgeQuery{
    query_set<Node>::match_values_in(set_of(ns)),
    query_set<Node>::matchall(),
  };

  std::unordered_map<Node, std::unordered_set<MultiDiEdge>> result =
      group_by(g.query_edges(query),
               [&](MultiDiEdge const &e) { return g.get_multidiedge_src(e); })
          .l_to_r();

  for (Node const &n : ns) {
    result[n];
  }

  return result;
}

} // namespace FlexFlow
