#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_RECORD_FORMATTER_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_RECORD_FORMATTER_H

#include <sstream>
#include <vector>
#include "utils/orientation.dtg.h"

namespace FlexFlow {

/**
 * \brief Helper interface for generating 
 * <a href="https://graphviz.org/doc/info/shapes.html#record">DOT/graphviz records</a>.
 *
 * \note This is very old code and should not be emulated stylistically.
 *
 * \see \ref DotFile
 * \see \ref mk_empty_record
 */
class RecordFormatter {
public:
  RecordFormatter() = delete;
  explicit RecordFormatter(Orientation, std::vector<std::string> const &pieces);

  friend RecordFormatter &operator<<(RecordFormatter &r,
                                     std::string const &tok);
  friend RecordFormatter &operator<<(RecordFormatter &r, int tok);
  friend RecordFormatter &operator<<(RecordFormatter &r, float tok);
  friend RecordFormatter &operator<<(RecordFormatter &r,
                                     RecordFormatter const &sub_r);
  friend RecordFormatter &operator<<(RecordFormatter &r,
                                     std::ostringstream &oss);
  friend std::ostream &operator<<(std::ostream &s, RecordFormatter const &r);

public:
  Orientation orientation;
  std::vector<std::string> pieces;
};

RecordFormatter mk_empty_record(Orientation);

}

#endif
