from flexflow.core import *
from flexflow.keras.datasets import cifar10
from flexflow.onnx.model import ONNXModel
import numpy as np

from accuracy import ModelAccuracy
from PIL import Image

MODEL_DIRECTORY = "/home/groups/aaiken/unger/models/inception"

def top_level_task():
  ffconfig = FFConfig()
  inceptionconfig = NetConfig()
  print(inceptionconfig.dataset_path)
  ffconfig.parse_args()
  print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" %(ffconfig.get_batch_size(), ffconfig.get_workers_per_node(), ffconfig.get_num_nodes()))
  ffmodel = FFModel(ffconfig)

  dims_input = [ffconfig.get_batch_size(), 3, 299, 299]
  input = ffmodel.create_tensor(dims_input, DataType.DT_FLOAT)

  onnx_model = ONNXModel(f"{MODEL_DIRECTORY}/inception_16_optimized.onnx")
  t = onnx_model.apply(ffmodel, {"data": input})
  t = ffmodel.pool2d(t, 8, 8, 1, 1, 0, 0, PoolType.POOL_AVG)
  t = ffmodel.flat(t)
  t = ffmodel.dense(t, 10)
  t = ffmodel.softmax(t)

  ffoptimizer = SGDOptimizer(ffmodel, 0.01)
  ffmodel.set_sgd_optimizer(ffoptimizer)
  ffmodel.compile(loss_type=LossType.LOSS_SPARSE_CATEGORICAL_CROSSENTROPY, metrics=[])
  label = ffmodel.get_label_tensor()

  num_samples = 10000

  #(x_train, y_train), (x_test, y_test) = cifar10.load_data(num_samples)

  full_input_np = np.empty((num_samples, 3, 299, 299), dtype=np.float32)

  for i in range(0, num_samples):
    # image = x_train[i, :, :, :]
    # image = image.transpose(1, 2, 0)
    # pil_image = Image.fromarray(image)
    # pil_image = pil_image.resize((299,299), Image.NEAREST)
    # image = np.array(pil_image, dtype=np.float32)
    # image = image.transpose(2, 0, 1)
    full_input_np[i, :, :, :] = np.random.rand(3, 299, 299)
    #full_input_np[i, :, :, :] = image

  full_input_np /= 255

  #y_train = y_train.astype('int32')
  #print(y_train.shape)
  y_train = np.random.randint(1000, size=(num_samples, 1))
  full_label_np = y_train

  dims_full_input = [num_samples, 3, 299, 299]
  full_input = ffmodel.create_tensor(dims_full_input, DataType.DT_FLOAT)

  dims_full_label = [num_samples, 1]
  full_label = ffmodel.create_tensor(dims_full_label, DataType.DT_INT32)

  full_input.attach_numpy_array(ffconfig, full_input_np)
  full_label.attach_numpy_array(ffconfig, full_label_np)

  dataloader_input = SingleDataLoader(ffmodel, input, full_input, num_samples, DataType.DT_FLOAT)
  dataloader_label = SingleDataLoader(ffmodel, label, full_label, num_samples, DataType.DT_INT32)

  full_input.detach_numpy_array(ffconfig)
  full_label.detach_numpy_array(ffconfig)

  num_samples = dataloader_input.get_num_samples()
  assert dataloader_input.get_num_samples() == dataloader_label.get_num_samples()

  ffmodel.init_layers()

  epochs = ffconfig.get_epochs()
  #epochs = 5

  ts_start = ffconfig.get_current_time()

  ffmodel.fit(x=dataloader_input, y=dataloader_label, batch_size=ffconfig.get_batch_size(), epochs=epochs)

  ts_end = ffconfig.get_current_time()
  run_time = 1e-6 * (ts_end - ts_start);
  print("epochs %d, ELAPSED TIME = %.4fs, THROUGHPUT = %.2f samples/s\n" %(epochs, run_time, num_samples * epochs / run_time));
  perf_metrics = ffmodel.get_perf_metrics()
  accuracy = perf_metrics.get_accuracy()
  #if accuracy < ModelAccuracy.CIFAR10_ALEXNET.value:
    #assert 0, 'Check Accuracy'

if __name__ == "__main__":
  print("inception onnx")
  top_level_task()
