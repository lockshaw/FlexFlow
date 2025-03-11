#ifndef _FLEXFLOW_KERNELS_INCLUDE_KERNELS_PERF_METRICS_H
#define _FLEXFLOW_KERNELS_INCLUDE_KERNELS_PERF_METRICS_H

#include "utils/fmt.h"
#include <optional>

namespace FlexFlow {

struct PerfMetrics {
public:
  PerfMetrics() = delete;
  explicit PerfMetrics(double start_time);
  PerfMetrics(int train_all,
              std::optional<int> train_correct,
              std::optional<float> cce_loss,
              std::optional<float> sparse_cce_loss,
              std::optional<float> mse_loss,
              std::optional<float> rmse_loss,
              std::optional<float> mae_loss,
              double start_time_micro,
              double current_time_micro);

  bool operator==(PerfMetrics const &) const;
  bool operator!=(PerfMetrics const &) const;

public:
  int train_all = 0;                    // measure_accuracy_denominator
  std::optional<int> train_correct = 0; // measure_accuracy numerator
  std::optional<float> cce_loss =
      std::nullopt; // measure_categorical_crossentropy
  std::optional<float> sparse_cce_loss =
      0.0f; // measure_sparse_categorical_crossentropy
  std::optional<float> mse_loss = 0.0f;  // measure_mean_squared_error
  std::optional<float> rmse_loss = 0.0f; // measure_root_mean_squared_error
  std::optional<float> mae_loss = 0.0f;  // measure_mean_absolute_error
  double start_time;
  double current_time;
private:
  std::tuple<
    decltype(train_all) const &,
    decltype(train_correct) const &,
    decltype(cce_loss) const &,
    decltype(sparse_cce_loss) const &,
    decltype(mse_loss) const &,
    decltype(rmse_loss) const &,
    decltype(mae_loss) const &,
    decltype(start_time) const &,
    decltype(current_time) const &,
  > tie() const;
};

float get_throughput(PerfMetrics const &);
float get_accuracy(PerfMetrics const &);

PerfMetrics update(PerfMetrics const &, PerfMetrics const &);
PerfMetrics apply_scale(PerfMetrics const &, float scale);

} // namespace FlexFlow

namespace fmt {

template <>
struct formatter<::FlexFlow::PerfMetrics> : formatter<std::string> {
  template <typename FormatContext>
  auto format(::FlexFlow::PerfMetrics const &m, FormatContext &ctx)
      -> decltype(ctx.out()) {
    auto out = fmt::memory_buffer();
    fmt::format_to(std::back_inserter(out), "PerfMetrics[");
    if (m.train_correct.has_value()) {
      fmt::format_to(std::back_inserter(out),
                     " accuracy={:.2f}%",
                     100.0 * get_accuracy(m));
    }
    if (m.cce_loss.has_value()) {
      fmt::format_to(
          std::back_inserter(out), " cce={:.2f}", m.cce_loss.value());
    }
    if (m.sparse_cce_loss.has_value()) {
      fmt::format_to(std::back_inserter(out),
                     " sparse_cce={:.2f}",
                     m.sparse_cce_loss.value());
    }
    if (m.mse_loss.has_value()) {
      fmt::format_to(
          std::back_inserter(out), " mse={:.2f}", m.mse_loss.value());
    }
    if (m.rmse_loss.has_value()) {
      fmt::format_to(
          std::back_inserter(out), " rmse={:.2f}", m.rmse_loss.value());
    }
    if (m.mae_loss.has_value()) {
      fmt::format_to(
          std::back_inserter(out), " mae={:.2f}", m.mae_loss.value());
    }
    fmt::format_to(
        std::back_inserter(out), "throughput={:.2f}", get_throughput(m));
    return formatter<std::string>::format(fmt::to_string(out), ctx);
  }
};

} // namespace fmt

#endif
