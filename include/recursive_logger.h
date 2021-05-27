#ifndef _FLEXFLOW_RECURSIVE_LOGGER_H
#define _FLEXFLOW_RECURSIVE_LOGGER_H

#include "legion/legion_utilities.h"
#include <functional>

enum class LogLevel {
  SPEW,
  DEBUG,
  INFO,
  PRINT,
  WARNING,
  ERROR
};

class ScopeMarker;

class RecursiveLogger {
public:
  /* RecursiveLogger(LegionRuntime::Logger::Category const &); */
  RecursiveLogger(std::string const &category_name);

  Realm::LoggerMessage log(LogLevel level);

  Realm::LoggerMessage error();
  Realm::LoggerMessage warning();
  Realm::LoggerMessage print();
  Realm::LoggerMessage info();
  Realm::LoggerMessage debug();
  Realm::LoggerMessage spew();
  void enter();
  void leave();
  ScopeMarker auto_enter();
  void nest(std::function<void()> const &f);
private:
  void format_message(Realm::LoggerMessage &);
private:
  int depth = 0;

  LegionRuntime::Logger::Category logger;

  friend class ScopeMarker;
};

class ScopeMarker {
public:
  ScopeMarker(RecursiveLogger &);
  ScopeMarker(RecursiveLogger &, int);
  ~ScopeMarker();

private:
  int saved;
  RecursiveLogger &logger;
};

#endif // _FLEXFLOW_RECURSIVE_LOGGER_H
