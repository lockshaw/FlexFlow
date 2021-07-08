#include "model.h"
#define MAX_NUM_SAMPLES 4196

using namespace Legion;
using namespace std;

struct SparseMLPConfig { };

class DataLoader {
public:
  DataLoader(FFModel& ff, const SparseMLPConfig& config,
             Tensor _input, Tensor _label);
  static void load_input(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx,
                         Runtime* runtime);
  static void load_label(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx,
                         Runtime* runtime);
  static void load_entire_dataset(const Task *task,
                                  const std::vector<PhysicalRegion> &regions,
                                  Context ctx,
                                  Runtime* runtime);
  void next_batch(FFModel&);
  void reset(void);
public:
  int num_samples, next_index;
  Tensor full_input, batch_input;
  Tensor full_label, batch_label;
};

struct SampleIdxs {
  int num_samples;
  int idxs[MAX_NUM_SAMPLES];
};

