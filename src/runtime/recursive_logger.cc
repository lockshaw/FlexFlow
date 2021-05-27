#include "recursive_logger.h"

RecursiveLogger::RecursiveLogger(std::string const &category_name)
  : logger(category_name)
{ }

void RecursiveLogger::format_message(Realm::LoggerMessage &msg) {
  msg << this->depth << " ";
  for (int i = 0; i < this->depth; i++) {
    msg << " ";
  }
}

Realm::LoggerMessage RecursiveLogger::log(LogLevel level) {
  switch (level) {
    case LogLevel::SPEW:
      return this->spew();
    case LogLevel::DEBUG:
      return this->debug();
    case LogLevel::INFO:
      return this->info();
    case LogLevel::PRINT:
      return this->print();
    case LogLevel::WARNING:
      return this->warning();
    case LogLevel::ERROR:
      return this->error();
    default:
      throw std::runtime_error("Unknown log level");
  }
}
Realm::LoggerMessage RecursiveLogger::error() {
  Realm::LoggerMessage msg = this->logger.error();
  format_message(msg);
  return msg;
}

Realm::LoggerMessage RecursiveLogger::warning() {
  Realm::LoggerMessage msg = this->logger.warning();
  format_message(msg);
  return msg;
}

Realm::LoggerMessage RecursiveLogger::print() {
  Realm::LoggerMessage msg = this->logger.print();
  format_message(msg);
  return msg;
}

Realm::LoggerMessage RecursiveLogger::info() {
  Realm::LoggerMessage msg = this->logger.info();
  format_message(msg);
  return msg;
}

Realm::LoggerMessage RecursiveLogger::debug() {
  Realm::LoggerMessage msg = this->logger.debug();
  format_message(msg);
  return msg;
}

Realm::LoggerMessage RecursiveLogger::spew() {
  Realm::LoggerMessage msg = this->logger.spew();
  format_message(msg);
  return msg;
}

ScopeMarker RecursiveLogger::auto_enter() {
  this->depth++;
  return ScopeMarker(*this, this->depth - 1);
}

void RecursiveLogger::enter() {
  this->depth++;
}

void RecursiveLogger::leave() { 
  this->depth--;
}

void RecursiveLogger::nest(std::function<void()> const &f) {
  auto scope = this->auto_enter();
  f();
}

ScopeMarker::ScopeMarker(RecursiveLogger &logger) 
  : ScopeMarker(logger, logger.depth)
{ }

ScopeMarker::ScopeMarker(RecursiveLogger &logger, int depth) 
  : logger(logger), saved(depth)
{ }

ScopeMarker::~ScopeMarker() {
  this->logger.depth = this->saved;
}
