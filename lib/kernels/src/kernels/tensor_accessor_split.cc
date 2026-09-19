#include "kernels/tensor_accessor_split.h"

namespace FlexFlow {

std::vector<GenericTensorAccessorW>
  tensor_accessor_split(GenericTensorAccessorR const &input,
                        ff_dim_t const &axis,
                        std::vector<positive_int> const &sizes,
                        Allocator &allocator)
{
  std::vector<TensorShape> output_shapes =
    transform(
      sizes,
      [&](positive_int size) -> TensorShape {
        TensorShape output_shape = input.shape;
        dim_at_idx(output_shape.dims, axis) = size;
        return output_shape;
      });

  std::vector<GenericTensorAccessorW> outputs =
    transform(output_shapes,
              [&](TensorShape const &output_shape) -> GenericTensorAccessorW {
                return allocator.allocate_tensor(output_shape);
              });

  tensor_accessor_split_to(
    input,
    axis,
    sizes,
    outputs);

  return outputs;
}

static std::pair<nonnegative_int, nonnegative_int>
      calculate_output_tensor_coord(nonnegative_int input_coord,
                                    std::vector<positive_int> const &output_sizes) {
  std::vector<nonnegative_int> output_sizes_nonnegative =
    transform(output_sizes,
              [](positive_int x) -> nonnegative_int {
                return x.nonnegative_int_from_positive_int();
              });

  auto add = [&](nonnegative_int accum, nonnegative_int x) -> nonnegative_int {
    return accum + x;
  };

  std::vector<nonnegative_int> points = scanl(output_sizes, 0_n, add);

  std::vector<std::pair<nonnegative_int, nonnegative_int>> intervals =
    zip_strict(
      slice(points, 0, -1),
      slice(points, 1, std::nullopt));

  for (auto const &[idx, interval] : enumerate(intervals)) {
    if (interval.first <= input_coord && input_coord < interval.second) {
      nonnegative_int in_tensor_coord = nonnegative_int{
        input_coord.int_from_nonnegative_int() - interval.first.int_from_nonnegative_int()
      };

      return std::pair{
        idx,
        in_tensor_coord,
      };
    }
  }

  PANIC();
}

template <DataType DT>
struct CPUSplitTensorAccessor {
  template <typename F>
  void operator()(GenericTensorAccessorR const &input,
                  ff_dim_t const &axis,
                  std::vector<GenericTensorAccessorW> const &outputs) {
    ASSERT(input.device_type == DeviceType::CPU);

    for (GenericTensorAccessorW const &output : outputs) {
      ASSERT(output.device_type == DeviceType::CPU);
    }

    std::vector<positive_int> output_tensor_sizes_in_axis =
      transform(outputs,
                [&](GenericTensorAccessorW const &output) -> positive_int {
                  return output.shape.dims.at(axis);
                });

    for (TensorDimsCoord const &coord :
         get_tensor_dims_coord_set(input.shape.dims)) {

      nonnegative_int axis_component = tensor_dims_coord_at_idx(coord, axis);

      std::pair<nonnegative_int, nonnegative_int> output_coord =
        calculate_output_tensor_coord(axis_component, output_tensor_sizes_in_axis);

      nonnegative_int output_idx = output_coord.first;
      nonnegative_int output_tensor_coord_component = output_coord.second;

      GenericTensorAccessorW output = outputs.at(output_idx.int_from_nonnegative_int());

      TensorDimsCoord output_coord = coord;
      tensor_dims_coord_at_idx(output_coord, axis) = output_tensor_coord_component;

      output.at<DT>(output_coord) = input.at<DT>(coord);
    }
  }
};

void
  tensor_accessor_split_to(GenericTensorAccessorR const &input,
                        ff_dim_t const &axis,
                        std::vector<positive_int> const &sizes,
                        std::vector<GenericTensorAccessorW> const &outputs)
{
  for (auto const &[size, output] : zip_strict(sizes, outputs)) {
    ASSERT(size == output.shape.dims.at(axis));
  }

  DataTypeDispatch1<CPUSplitTensorAccessor>{}(
      input.shape.data_type, input, axis, sizes, outputs);
}

} // namespace FlexFlow
