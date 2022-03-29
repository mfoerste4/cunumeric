/* Copyright 2022 NVIDIA Corporation
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
 *
 */
#include "cunumeric/sort/utilities.h"
#include "cunumeric/sort/utilities.inl"

#include <thrust/device_vector.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>

namespace cunumeric {

using namespace Legion;

/*
  BOOL_LT         = LEGION_TYPE_BOOL,
  INT8_LT         = LEGION_TYPE_INT8,
  INT16_LT        = LEGION_TYPE_INT16,
  INT32_LT        = LEGION_TYPE_INT32,
  INT64_LT        = LEGION_TYPE_INT64,
  UINT8_LT        = LEGION_TYPE_UINT8,
  UINT16_LT       = LEGION_TYPE_UINT16,
  UINT32_LT       = LEGION_TYPE_UINT32,
  UINT64_LT       = LEGION_TYPE_UINT64,
  HALF_LT         = LEGION_TYPE_FLOAT16,
  FLOAT_LT        = LEGION_TYPE_FLOAT32,
  DOUBLE_LT       = LEGION_TYPE_FLOAT64,
  COMPLEX64_LT    = LEGION_TYPE_COMPLEX64,
  COMPLEX128_LT   = LEGION_TYPE_COMPLEX128
  */
template void cub_local_sort<legate_type_of<LegateTypeCode::BOOL_LT>>(
  const legate_type_of<LegateTypeCode::BOOL_LT>*,
  legate_type_of<LegateTypeCode::BOOL_LT>*,
  const int64_t*,
  int64_t*,
  const size_t,
  const size_t,
  cudaStream_t);
template void cub_local_sort<legate_type_of<LegateTypeCode::INT8_LT>>(
  const legate_type_of<LegateTypeCode::INT8_LT>*,
  legate_type_of<LegateTypeCode::INT8_LT>*,
  const int64_t*,
  int64_t*,
  const size_t,
  const size_t,
  cudaStream_t);
template void cub_local_sort<legate_type_of<LegateTypeCode::INT16_LT>>(
  const legate_type_of<LegateTypeCode::INT16_LT>*,
  legate_type_of<LegateTypeCode::INT16_LT>*,
  const int64_t*,
  int64_t*,
  const size_t,
  const size_t,
  cudaStream_t);
template void cub_local_sort<legate_type_of<LegateTypeCode::INT32_LT>>(
  const legate_type_of<LegateTypeCode::INT32_LT>*,
  legate_type_of<LegateTypeCode::INT32_LT>*,
  const int64_t*,
  int64_t*,
  const size_t,
  const size_t,
  cudaStream_t);
template void cub_local_sort<legate_type_of<LegateTypeCode::INT64_LT>>(
  const legate_type_of<LegateTypeCode::INT64_LT>*,
  legate_type_of<LegateTypeCode::INT64_LT>*,
  const int64_t*,
  int64_t*,
  const size_t,
  const size_t,
  cudaStream_t);
template void cub_local_sort<legate_type_of<LegateTypeCode::UINT8_LT>>(
  const legate_type_of<LegateTypeCode::UINT8_LT>*,
  legate_type_of<LegateTypeCode::UINT8_LT>*,
  const int64_t*,
  int64_t*,
  const size_t,
  const size_t,
  cudaStream_t);
template void cub_local_sort<legate_type_of<LegateTypeCode::UINT16_LT>>(
  const legate_type_of<LegateTypeCode::UINT16_LT>*,
  legate_type_of<LegateTypeCode::UINT16_LT>*,
  const int64_t*,
  int64_t*,
  const size_t,
  const size_t,
  cudaStream_t);
template void cub_local_sort<legate_type_of<LegateTypeCode::UINT32_LT>>(
  const legate_type_of<LegateTypeCode::UINT32_LT>*,
  legate_type_of<LegateTypeCode::UINT32_LT>*,
  const int64_t*,
  int64_t*,
  const size_t,
  const size_t,
  cudaStream_t);
template void cub_local_sort<legate_type_of<LegateTypeCode::UINT64_LT>>(
  const legate_type_of<LegateTypeCode::UINT64_LT>*,
  legate_type_of<LegateTypeCode::UINT64_LT>*,
  const int64_t*,
  int64_t*,
  const size_t,
  const size_t,
  cudaStream_t);
template void cub_local_sort<legate_type_of<LegateTypeCode::HALF_LT>>(
  const legate_type_of<LegateTypeCode::HALF_LT>*,
  legate_type_of<LegateTypeCode::HALF_LT>*,
  const int64_t*,
  int64_t*,
  const size_t,
  const size_t,
  cudaStream_t);
template void cub_local_sort<legate_type_of<LegateTypeCode::FLOAT_LT>>(
  const legate_type_of<LegateTypeCode::FLOAT_LT>*,
  legate_type_of<LegateTypeCode::FLOAT_LT>*,
  const int64_t*,
  int64_t*,
  const size_t,
  const size_t,
  cudaStream_t);
template void cub_local_sort<legate_type_of<LegateTypeCode::DOUBLE_LT>>(
  const legate_type_of<LegateTypeCode::DOUBLE_LT>*,
  legate_type_of<LegateTypeCode::DOUBLE_LT>*,
  const int64_t*,
  int64_t*,
  const size_t,
  const size_t,
  cudaStream_t);
// template void cub_local_sort<legate_type_of<LegateTypeCode::COMPLEX64_LT>>(const
// legate_type_of<LegateTypeCode::COMPLEX64_LT>*, legate_type_of<LegateTypeCode::COMPLEX64_LT>*,
// const int64_t*, int64_t*, const size_t, const size_t, cudaStream_t); template void
// cub_local_sort<legate_type_of<LegateTypeCode::COMPLEX128_LT>>(const
// legate_type_of<LegateTypeCode::COMPLEX128_LT>*, legate_type_of<LegateTypeCode::COMPLEX128_LT>*,
// const int64_t*, int64_t*, const size_t, const size_t, cudaStream_t);

template <class VAL>
void cub_local_sort(const VAL* values_in,
                    VAL* values_out,
                    const int64_t* indices_in,
                    int64_t* indices_out,
                    const size_t volume,
                    const size_t sort_dim_size,
                    cudaStream_t stream)
{
  Buffer<VAL> keys_in;
  const VAL* values_in_cub = values_in;
  if (values_in == values_out) {
    keys_in       = create_buffer<VAL>(volume, Legion::Memory::Kind::GPU_FB_MEM);
    values_in_cub = keys_in.ptr(0);
    CHECK_CUDA(cudaMemcpyAsync(
      keys_in.ptr(0), values_out, sizeof(VAL) * volume, cudaMemcpyDeviceToDevice, stream));
  }

  size_t temp_storage_bytes = 0;
  if (indices_out == nullptr) {
    if (volume == sort_dim_size) {
      // sort (initial call to compute buffer size)
      cub::DeviceRadixSort::SortKeys(
        nullptr, temp_storage_bytes, values_in_cub, values_out, volume, 0, sizeof(VAL) * 8, stream);
      auto temp_storage =
        create_buffer<unsigned char>(temp_storage_bytes, Legion::Memory::Kind::GPU_FB_MEM);
      cub::DeviceRadixSort::SortKeys(temp_storage.ptr(0),
                                     temp_storage_bytes,
                                     values_in_cub,
                                     values_out,
                                     volume,
                                     0,
                                     sizeof(VAL) * 8,
                                     stream);
      temp_storage.destroy();
    } else {
      // segmented sort (initial call to compute buffer size)
      // generate start/end positions for all segments via iterators to avoid allocating buffers
      auto off_start_pos_it =
        thrust::make_transform_iterator(thrust::make_counting_iterator(0), multiply(sort_dim_size));
      auto off_end_pos_it =
        thrust::make_transform_iterator(thrust::make_counting_iterator(1), multiply(sort_dim_size));

      cub::DeviceSegmentedRadixSort::SortKeys(nullptr,
                                              temp_storage_bytes,
                                              values_in_cub,
                                              values_out,
                                              volume,
                                              volume / sort_dim_size,
                                              off_start_pos_it,
                                              off_end_pos_it,
                                              0,
                                              sizeof(VAL) * 8,
                                              stream);
      auto temp_storage =
        create_buffer<unsigned char>(temp_storage_bytes, Legion::Memory::Kind::GPU_FB_MEM);

      cub::DeviceSegmentedRadixSort::SortKeys(temp_storage.ptr(0),
                                              temp_storage_bytes,
                                              values_in_cub,
                                              values_out,
                                              volume,
                                              volume / sort_dim_size,
                                              off_start_pos_it,
                                              off_end_pos_it,
                                              0,
                                              sizeof(VAL) * 8,
                                              stream);
      temp_storage.destroy();
    }
  } else {
    Buffer<int64_t> idx_in;
    const int64_t* indices_in_cub = indices_in;
    if (indices_in == indices_out) {
      idx_in         = create_buffer<int64_t>(volume, Legion::Memory::Kind::GPU_FB_MEM);
      indices_in_cub = idx_in.ptr(0);
      CHECK_CUDA(cudaMemcpyAsync(
        idx_in.ptr(0), indices_out, sizeof(int64_t) * volume, cudaMemcpyDeviceToDevice, stream));
    }

    if (volume == sort_dim_size) {
      // argsort (initial call to compute buffer size)
      cub::DeviceRadixSort::SortPairs(nullptr,
                                      temp_storage_bytes,
                                      values_in_cub,
                                      values_out,
                                      indices_in_cub,
                                      indices_out,
                                      volume,
                                      0,
                                      sizeof(VAL) * 8,
                                      stream);

      auto temp_storage =
        create_buffer<unsigned char>(temp_storage_bytes, Legion::Memory::Kind::GPU_FB_MEM);

      cub::DeviceRadixSort::SortPairs(temp_storage.ptr(0),
                                      temp_storage_bytes,
                                      values_in_cub,
                                      values_out,
                                      indices_in_cub,
                                      indices_out,
                                      volume,
                                      0,
                                      sizeof(VAL) * 8,
                                      stream);
      temp_storage.destroy();
    } else {
      // segmented argsort (initial call to compute buffer size)
      // generate start/end positions for all segments via iterators to avoid allocating buffers
      auto off_start_pos_it =
        thrust::make_transform_iterator(thrust::make_counting_iterator(0), multiply(sort_dim_size));
      auto off_end_pos_it =
        thrust::make_transform_iterator(thrust::make_counting_iterator(1), multiply(sort_dim_size));

      cub::DeviceSegmentedRadixSort::SortPairs(nullptr,
                                               temp_storage_bytes,
                                               values_in_cub,
                                               values_out,
                                               indices_in_cub,
                                               indices_out,
                                               volume,
                                               volume / sort_dim_size,
                                               off_start_pos_it,
                                               off_end_pos_it,
                                               0,
                                               sizeof(VAL) * 8,
                                               stream);

      auto temp_storage =
        create_buffer<unsigned char>(temp_storage_bytes, Legion::Memory::Kind::GPU_FB_MEM);

      cub::DeviceSegmentedRadixSort::SortPairs(temp_storage.ptr(0),
                                               temp_storage_bytes,
                                               values_in_cub,
                                               values_out,
                                               indices_in_cub,
                                               indices_out,
                                               volume,
                                               volume / sort_dim_size,
                                               off_start_pos_it,
                                               off_end_pos_it,
                                               0,
                                               sizeof(VAL) * 8,
                                               stream);
      temp_storage.destroy();
    }
    if (indices_in == indices_out) idx_in.destroy();
  }

  if (values_in == values_out) keys_in.destroy();
}

}  // namespace cunumeric