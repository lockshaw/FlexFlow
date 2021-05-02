from flexflow.core import *
# from flexflow.keras.datasets import cifar10
from flexflow.onnx.model import ONNXModel
import logging
import os

from accuracy import ModelAccuracy
# from PIL import Image

logging.basicConfig(level=logging.DEBUG)

MODEL_DIRECTORY = "/home/unger/models/resnext50/"

def top_level_task():
  ffconfig = FFConfig()
  print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" %(ffconfig.batch_size, ffconfig.workers_per_node, ffconfig.num_nodes))
  ffmodel = FFModel(ffconfig)

  dims_input = [ffconfig.batch_size, 3, 224, 224]
  input = ffmodel.create_tensor(dims_input, DataType.DT_FLOAT)

  #normal_cells = os.getenv("NORMAL_CELLS")
  model_tag = os.getenv("MODEL_TAG")
  use_optimized = bool(int(os.getenv("OPTIMIZED")))
  print(os.getenv("OPTIMIZED"), use_optimized)
  optimized_suffix = 'optimized' if use_optimized else 'unoptimized'
  #onnx_filename = f"{MODEL_DIRECTORY}/resnext50{model_tag}_{ffconfig.batch_size}_{optimized_suffix}.onnx"
  onnx_filename = f"{MODEL_DIRECTORY}/resnext50{model_tag}_32_{optimized_suffix}.onnx"
  onnx_model = ONNXModel(onnx_filename)
  # onnx_model = ONNXModel(f"{MODEL_DIRECTORY}/nasneta_{ffconfig.get_batch_size()}_optimized_n5.onnx")
  #onnx_model = ONNXModel(f"{MODEL_DIRECTORY}/resnext50_{ffconfig.get_batch_size()}_optimized.onnx")
  t = onnx_model.apply(ffmodel, {"data": input})
  t = ffmodel.relu(t, name='HEYA1')
  t = ffmodel.pool2d(t, t.dims[2], t.dims[3], 1, 1, 0, 0, pool_type=PoolType.POOL_AVG, name='HEYA2')
  t = ffmodel.flat(t, name='HEYA3')
  t = ffmodel.dense(t, 1000, name='HEYA4')
  t = ffmodel.softmax(t, name='HEYA5')

  ffoptimizer = SGDOptimizer(ffmodel, 0.01)
  ffmodel.optimizer = ffoptimizer
  ffmodel.compile(loss_type=LossType.LOSS_SPARSE_CATEGORICAL_CROSSENTROPY, metrics=[MetricsType.METRICS_ACCURACY, MetricsType.METRICS_SPARSE_CATEGORICAL_CROSSENTROPY])
  label = ffmodel.label_tensor

  num_samples = 1000

  # for i in range(0, num_samples):
  #   image = x_train[i, :, :, :]
  #   image = image.transpose(1, 2, 0)
  #   pil_image = Image.fromarray(image)
  #   pil_image = pil_image.resize((224,224), Image.NEAREST)
  #   image = np.array(pil_image, dtype=np.float32)
  #   image = image.transpose(2, 0, 1)
  #   full_input_np[i, :, :, :] = image

  full_input_np = np.random.rand(num_samples, 3, 224, 224)
  full_label_np = np.random.rand(num_samples, 1)

  dims_full_input = [num_samples, 3, 224, 224]
  full_input = ffmodel.create_tensor(dims_full_input, DataType.DT_FLOAT)

  dims_full_label = [num_samples, 1]
  full_label = ffmodel.create_tensor(dims_full_label, DataType.DT_INT32)

  full_input.attach_numpy_array(ffconfig, full_input_np)
  full_label.attach_numpy_array(ffconfig, full_label_np)

  print('LABEL 1')
  dataloader_input = SingleDataLoader(ffmodel, input, full_input, num_samples, DataType.DT_FLOAT)
  dataloader_label = SingleDataLoader(ffmodel, label, full_label, num_samples, DataType.DT_INT32)

  print('LABEL 2')
  full_input.detach_numpy_array(ffconfig)
  full_label.detach_numpy_array(ffconfig)

  print('LABEL 3')
  num_samples = dataloader_input.num_samples
  assert dataloader_input.num_samples == dataloader_label.num_samples

  print('LABEL 4')
  ffmodel.init_layers()

  print('LABEL 5')
  epochs = ffconfig.epochs

  ts_start = ffconfig.get_current_time()

  print('LABEL 6')
  ffmodel.fit(x=dataloader_input, y=dataloader_label, batch_size=ffconfig.batch_size, epochs=epochs)

  print('LABEL 7')
  ts_end = ffconfig.get_current_time()
  print('LABEL 8')
  run_time = 1e-6 * (ts_end - ts_start);
  print("epochs %d, ELAPSED TIME = %.4fs, THROUGHPUT = %.2f samples/s\n" %(epochs, run_time, num_samples * epochs / run_time));
  perf_metrics = ffmodel.get_perf_metrics()
  accuracy = perf_metrics.get_accuracy()
  #if accuracy < ModelAccuracy.CIFAR10_ALEXNET.value:
    #assert 0, 'Check Accuracy'

if __name__ == "__main__":
  print("resnet onnx")
  top_level_task()
