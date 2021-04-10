/* Copyright 2019 Stanford
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "model.h"


/* NOTE: This is the very, very un-performant.
- You should at least use Eigen or uBlas or so for matrix multiplications.
  Preferably implement on GPU */

Tensor FFModel::aggregate_spec(const Tensor* inputs, /* gate_preds, gate_assign, whole_gate_preds, n * exp_pred */
                          int n, float lambda_bal, const char* name)
{
  AggregateSpec* aggr = new AggregateSpec(*this, inputs, n, lambda_bal, name);
  layers.push_back(aggr);
  return aggr->outputs[0];
}


AggregateSpec::AggregateSpec(FFModel& model,
                    const Tensor* _inputs,
                    int _n, float _lambda_bal, const char* name)
: Op(model, OP_AGG_SPEC, name, _n+3, _inputs),
  n(_n), lambda_bal(_lambda_bal)
  //profiling(model.config.profiling)
{
  assert(inputs[0].numDim == 2); // TODO: More flexible. pass in images etc.
  assert(inputs[1].numDim == 2);
  assert(inputs[0].adim[0] == inputs[1].adim[0]);
  assert(n == inputs[2].adim[0]);
  assert(inputs[0].adim[1] == inputs[1].adim[1]);
  assert(inputs[0].adim[1] == inputs[2].adim[1]);
  assert(n+3 == numInputs);
  assert(n > 0);

  int out_dim = inputs[3].adim[0];
  int batch_size = inputs[0].adim[1];
  int k = inputs[0].adim[0];
  outputs[0].numDim = 2;
  outputs[0].adim[0] = out_dim;
  outputs[0].adim[1] = k*batch_size;

  for(int i = 1; i < n; i++) {
    assert(inputs[i+3].adim[0] == out_dim);
  }

  numWeights = 0;
}


void AggregateSpec::create_weights(FFModel& model)
{
  // Do nothing
}


void AggregateSpec::create_output_and_partition(FFModel& model)
{
  // Retrieve the task indexspace for the op
  std::string pcname = name;
  task_is = IndexSpaceT<2>(model.get_or_create_task_is(2, pcname));
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Rect<2> part_rect = runtime->get_index_space_domain(ctx, task_is);

  // Can only partition over the sample dim
  assert(part_rect.hi[0] == part_rect.lo[0]);

  int batch_size = inputs[0].adim[1];
  int out_dim = inputs[3].adim[0];
  int k = inputs[0].adim[0];

  const int dims[2] = {k*batch_size, out_dim};
  outputs[0] = model.create_tensor<2>(dims, DT_FLOAT, this);
  outputs[0].owner_op = this;
  outputs[0].owner_idx = 0;


  // Compute partition bound for input
  for(int i = 0; i < n+3; i++) {
    Rect<2> input_rect = runtime->get_index_partition_color_space(
        ctx, inputs[i].part.get_index_partition());
    if (input_rect == part_rect) {
      input_lps[i] = inputs[i].part;
      input_grad_lps[i] = inputs[i].part_grad;
    } else {
      model.create_disjoint_partition<2>(
        inputs[i], (IndexSpaceT<2>)task_is, input_lps[i], input_grad_lps[i]);
    }
  }
}


OpMeta* AggregateSpec::init_task(const Task* task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime* runtime)
{
  return NULL;
}


/* TODO: ?? . Also produces warning [warning 1071] LEGION WARNING: Region
requirement 7 of operation Aggregate Init Task (UID 140) in parent task
top_level (UID 1) is using uninitialized data for field(s) 0 of logical
region (81,28,53) */
void AggregateSpec::init(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexLauncher launcher(AGG_SPEC_INIT_TASK_ID, task_is,
                         TaskArgument(this, sizeof(AggregateSpec)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  // gate_preds
  launcher.add_region_requirement(
    RegionRequirement(input_lps[0], 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, inputs[0].region));
  launcher.add_field(0, FID_DATA);
  // gate_assign
  launcher.add_region_requirement(
    RegionRequirement(input_lps[1], 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, inputs[1].region));
  launcher.add_field(1, FID_DATA);
  // gate_assign_whole
  launcher.add_region_requirement(
    RegionRequirement(input_lps[2], 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, inputs[2].region));
  launcher.add_field(2, FID_DATA);
  // exp_preds
  for(int i = 0; i < n; i++) {
    launcher.add_region_requirement(
      RegionRequirement(input_lps[i+3], 0/*projection id*/,
        READ_WRITE, EXCLUSIVE, inputs[i+3].region));
    launcher.add_field(i+3, FID_DATA);
  }
  // output
  launcher.add_region_requirement(
    RegionRequirement(outputs[0].part, 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, outputs[0].region));
  launcher.add_field(n+3, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}


void agg_spec_forward(float** exp_preds,
        const int* exp_assign,
        const float* gating_net_preds,
        float* output,
        int n, // num experts
        int k, // num chosen experts
        int exp_samples, // max samples per expert
        int batch_size,
        int out_dim)
{
  std::vector<int> expert_idx(n, 0);
  for(int i = 0; i < batch_size; i++) {
    for(int j = 0; j < k; j++) {
      // Get pointer to chosen expert predictions
      int expert = exp_assign[k*i + j];
      if(expert_idx[expert] < exp_samples) { // sample not dropped
        float* chosen_exp_preds = exp_preds[expert] + expert_idx[expert]*out_dim;
        memcpy(output + i*k*out_dim + j*out_dim, chosen_exp_preds, out_dim*sizeof(float));
      }
      expert_idx[expert]++;
    }
  }
}


void agg_spec_backward(float** exp_grads,
        const int* exp_assign,
        const float* gating_net_preds,
        float* full_gating_grads,
        const float* output_grads,
        int n, // num experts
        int k, // num chosen experts
        int exp_samples, // max samples per expert
        float lambda_bal,
        int batch_size,
        int out_dim)
{
  std::vector<int> expert_idx(n, 0);
  std::vector<int> expert_assign(n, 0);
  for(int i = 0; i < batch_size; i++) {
    const float* sample_gating_weights = gating_net_preds + i*k;
    const float* sample_output_grads = output_grads + i*k*out_dim;
    float* sample_full_gate_grads = full_gating_grads + i*n;

    std::vector<float> sq_exp_errors(k, 0.0f);
    for(int j = 0; j < k; j++) {
      int expert = exp_assign[k*i + j];
      expert_assign[expert]++;

      if(expert_idx[expert] < exp_samples) { // sample not dropped
        // expert gradient
        float* chosen_exp_grad = exp_grads[expert] + expert_idx[expert]*out_dim;
        for(int l = 0; l < out_dim; l++) {
          chosen_exp_grad[l] += sample_gating_weights[j] * sample_output_grads[j*out_dim+l];
        }

        // get expert errors (squared L2 norm of gradient)
        for(int l = 0; l < out_dim; l++) {
          /* NOTE: sample output grads are per default normalized by 1/batch_size.
          So in the default case, multiplying *batch_size will make gating net
          grads normalized by 1/batchsize (and not 1/batch_size^2).*/
          sample_full_gate_grads[expert] += batch_size * sample_output_grads[j*out_dim+l] * sample_output_grads[j*out_dim+l];
        }

        expert_idx[expert]++;
      }
    }
  }

  // gating grads: balance term and enforce zero mean
  lambda_bal *= ((float)n)/batch_size;
  for(int i = 0; i < batch_size; i++) {
    float* sample_full_gate_grads = full_gating_grads + i*n;
    float grad_sum = 0.0f;
    for(int j = 0; j < n; j++) {
      sample_full_gate_grads[j] += lambda_bal*exp_assign[j];
      grad_sum += sample_full_gate_grads[j];
    }
    grad_sum /= n;
    for(int j = 0; j < n; j++) {
      sample_full_gate_grads[j] -= grad_sum;
    }
  }
}


void AggregateSpec::forward_task(const Task *task,
                             const std::vector<PhysicalRegion>& regions,
                             Context ctx, Runtime* runtime)
{
  int n = ((AggregateSpec*)task->args)->n;

  assert((int)regions.size() == n+3);
  assert((int)task->regions.size() == n+3);

  // get gate_pred, gate_assign, output
  const AccessorRO<float, 2> acc_gate_pred(regions[0], FID_DATA);
  const AccessorRO<int, 2> acc_gate_assign(regions[1], FID_DATA);
  const AccessorWO<float, 2> acc_output(regions[n+2], FID_DATA);

  Rect<2> rect_gate_pred = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<2> rect_gate_assign = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Rect<2> rect_output = runtime->get_index_space_domain(
      ctx, task->regions[n+2].region.get_index_space());

  coord_t batch_size = rect_gate_pred.hi[1] - rect_gate_pred.lo[1] + 1;
  assert(batch_size == rect_gate_assign.hi[1] - rect_gate_assign.lo[1] + 1);
  coord_t k = rect_gate_pred.hi[0] - rect_gate_pred.lo[0] + 1;
  assert(k == rect_gate_assign.hi[0] - rect_gate_assign.lo[0] + 1);
  assert(k*batch_size == rect_output.hi[1] - rect_output.lo[1] + 1);
  coord_t out_dim = rect_output.hi[0] - rect_output.lo[0] + 1;

  // get exp_preds
  float* exp_preds[n];
  // get first exp_pred and row and out_dim
  Domain exp_domain = runtime->get_index_space_domain(
    ctx, task->regions[2].region.get_index_space());
  exp_preds[0] = helperGetTensorPointerWO<float>(
    regions[2], task->regions[2], FID_DATA, ctx, runtime);
  coord_t rows = exp_domain.hi()[1] - exp_domain.lo()[1] + 1;
  assert(out_dim == exp_domain.hi()[0] - exp_domain.lo()[0] + 1);

  for(int i = 1; i < n; i++) {
    exp_domain = runtime->get_index_space_domain(
      ctx, task->regions[i+2].region.get_index_space());
    exp_preds[i] = helperGetTensorPointerWO<float>(
      regions[i+2], task->regions[i+2], FID_DATA, ctx, runtime);

    assert(rows == exp_domain.hi()[1] - exp_domain.lo()[1] + 1);
    assert(out_dim == exp_domain.hi()[0] - exp_domain.lo()[0] + 1);
  }

  agg_spec_forward(exp_preds, acc_gate_assign.ptr(rect_gate_assign),
    acc_gate_pred.ptr(rect_gate_pred), acc_output.ptr(rect_output), n, (int)k,
    rows, batch_size, out_dim);
}

void AggregateSpec::backward_task(const Task *task,
                              const std::vector<PhysicalRegion>& regions,
                              Context ctx, Runtime* runtime)
{
  int n = ((AggregateSpec*)task->args)->n;
  float lambda_bal = ((AggregateSpec*)task->args)->lambda_bal;

  assert((int)regions.size() == n+4);
  assert((int)task->regions.size() == n+4);

  // get gate_pred, gate_assin, full_gate_grad, output_grad
  const AccessorRO<float, 2> acc_gate_pred(regions[0], FID_DATA);
  const AccessorRO<int, 2> acc_gate_assign(regions[1], FID_DATA);
  const AccessorWO<float, 2> acc_full_gate_grad(regions[2], FID_DATA);
  const AccessorRO<float, 2> acc_output_grad(regions[n+3], FID_DATA);

  Rect<2> rect_gate_pred = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<2> rect_gate_assign = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Rect<2> rect_full_gate_grad = runtime->get_index_space_domain(
          ctx, task->regions[2].region.get_index_space());
  Rect<2> rect_out_grad = runtime->get_index_space_domain(
      ctx, task->regions[n+3].region.get_index_space());

  coord_t batch_size = rect_gate_pred.hi[1] - rect_gate_pred.lo[1] + 1;
  assert(batch_size == rect_gate_assign.hi[1] - rect_gate_assign.lo[1] + 1);
  assert(batch_size == rect_full_gate_grad.hi[1] - rect_full_gate_grad.lo[1] + 1);
  coord_t k = rect_gate_assign.hi[0] - rect_gate_assign.lo[0] + 1;
  assert(k*batch_size == rect_out_grad.hi[1] - rect_out_grad.lo[1] + 1);
  assert(rect_gate_pred.hi[0] - rect_gate_pred.lo[0] + 1 == k);
  coord_t out_dim = rect_out_grad.hi[0] - rect_out_grad.lo[0] + 1;
  assert(n == rect_full_gate_grad.hi[0] - rect_full_gate_grad.lo[0] + 1);

  // get exp_preds
  float* exp_grads[n];
  // get first exp_pred and row
  Domain exp_domain = runtime->get_index_space_domain(
    ctx, task->regions[3].region.get_index_space());
  exp_grads[0] = helperGetTensorPointerRW<float>(
    regions[3], task->regions[3], FID_DATA, ctx, runtime);
  coord_t rows = exp_domain.hi()[1] - exp_domain.lo()[1] + 1;
  assert(out_dim == exp_domain.hi()[0] - exp_domain.lo()[0] + 1);

  for(int i = 1; i < n; i++) {
    exp_domain = runtime->get_index_space_domain(
      ctx, task->regions[i+3].region.get_index_space());
    exp_grads[i] = helperGetTensorPointerRW<float>(
      regions[i+3], task->regions[i+3], FID_DATA, ctx, runtime);
    assert(rows == exp_domain.hi()[1] - exp_domain.lo()[1] + 1);
    assert(out_dim == exp_domain.hi()[0] - exp_domain.lo()[0] + 1);
  }

  agg_spec_backward(exp_grads, acc_gate_assign.ptr(rect_gate_assign),
    acc_gate_pred.ptr(rect_gate_pred), acc_full_gate_grad.ptr(rect_full_gate_grad),
    acc_output_grad.ptr(rect_out_grad), n, k, rows, lambda_bal, batch_size, out_dim);
}


void AggregateSpec::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexLauncher launcher(AGG_SPEC_FWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(AggregateSpec)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));

  // gate_preds
  launcher.add_region_requirement(
    RegionRequirement(input_lps[0], 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, inputs[0].region));
  launcher.add_field(0, FID_DATA);
  // gate_assign
  launcher.add_region_requirement(
    RegionRequirement(input_lps[1], 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, inputs[1].region));
  launcher.add_field(1, FID_DATA);

  // exp_preds
  for(int i = 0; i < n; i++) {
    launcher.add_region_requirement(
      RegionRequirement(input_lps[i+3], 0/*projection id*/,
        READ_WRITE, EXCLUSIVE, inputs[i+3].region));
    launcher.add_field(i+2, FID_DATA);
  }
  // output
  launcher.add_region_requirement(
    RegionRequirement(outputs[0].part, 0/*projection id*/,
      WRITE_ONLY, EXCLUSIVE, outputs[0].region));
  launcher.add_field(n+2, FID_DATA);

  runtime->execute_index_space(ctx, launcher);
}


void AggregateSpec::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexLauncher launcher(AGG_SPEC_BWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(AggregateSpec)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));

  // gate_preds
  launcher.add_region_requirement(
    RegionRequirement(input_lps[0], 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, inputs[0].region));
  launcher.add_field(0, FID_DATA);

  // gate_assign
  launcher.add_region_requirement(
    RegionRequirement(input_lps[1], 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, inputs[1].region));
  launcher.add_field(1, FID_DATA);

  // gate gradients full
  launcher.add_region_requirement(
    RegionRequirement(input_grad_lps[2], 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, inputs[2].region_grad));
  launcher.add_field(2, FID_DATA);

  // exp gradients
  for(int i = 0; i < n; i++) {
    launcher.add_region_requirement(
      RegionRequirement(input_grad_lps[i+3], 0/*projection id*/,
        READ_WRITE, EXCLUSIVE, inputs[i+3].region_grad));
    launcher.add_field(i+3, FID_DATA);
  }

  // output
  launcher.add_region_requirement(
    RegionRequirement(outputs[0].part_grad, 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, outputs[0].region_grad));
  launcher.add_field(n+3, FID_DATA);

  runtime->execute_index_space(ctx, launcher);
}


bool AggregateSpec::measure_operator_cost(Simulator* sim,
                                 const ParallelConfig& pc,
                                 CostMetrics& cost_metrics)
{
  //TODO: implement
  cost_metrics.forward_time = 0.0f;
  cost_metrics.backward_time = 0.0f;
  cost_metrics.memory_requirement = 0;
  return false;
}