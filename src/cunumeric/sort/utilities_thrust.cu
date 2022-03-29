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

#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/binary_search.h>
#include <thrust/tuple.h>
#include <thrust/execution_policy.h>

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
template void thrust_local_sort<legate_type_of<LegateTypeCode::BOOL_LT>>(
  const legate_type_of<LegateTypeCode::BOOL_LT>*,
  legate_type_of<LegateTypeCode::BOOL_LT>*,
  const int64_t*,
  int64_t*,
  const size_t,
  const size_t,
  const bool,
  cudaStream_t);
template void thrust_local_sort<legate_type_of<LegateTypeCode::INT8_LT>>(
  const legate_type_of<LegateTypeCode::INT8_LT>*,
  legate_type_of<LegateTypeCode::INT8_LT>*,
  const int64_t*,
  int64_t*,
  const size_t,
  const size_t,
  const bool,
  cudaStream_t);
template void thrust_local_sort<legate_type_of<LegateTypeCode::INT16_LT>>(
  const legate_type_of<LegateTypeCode::INT16_LT>*,
  legate_type_of<LegateTypeCode::INT16_LT>*,
  const int64_t*,
  int64_t*,
  const size_t,
  const size_t,
  const bool,
  cudaStream_t);
template void thrust_local_sort<legate_type_of<LegateTypeCode::INT32_LT>>(
  const legate_type_of<LegateTypeCode::INT32_LT>*,
  legate_type_of<LegateTypeCode::INT32_LT>*,
  const int64_t*,
  int64_t*,
  const size_t,
  const size_t,
  const bool,
  cudaStream_t);
template void thrust_local_sort<legate_type_of<LegateTypeCode::INT64_LT>>(
  const legate_type_of<LegateTypeCode::INT64_LT>*,
  legate_type_of<LegateTypeCode::INT64_LT>*,
  const int64_t*,
  int64_t*,
  const size_t,
  const size_t,
  const bool,
  cudaStream_t);
template void thrust_local_sort<legate_type_of<LegateTypeCode::UINT8_LT>>(
  const legate_type_of<LegateTypeCode::UINT8_LT>*,
  legate_type_of<LegateTypeCode::UINT8_LT>*,
  const int64_t*,
  int64_t*,
  const size_t,
  const size_t,
  const bool,
  cudaStream_t);
template void thrust_local_sort<legate_type_of<LegateTypeCode::UINT16_LT>>(
  const legate_type_of<LegateTypeCode::UINT16_LT>*,
  legate_type_of<LegateTypeCode::UINT16_LT>*,
  const int64_t*,
  int64_t*,
  const size_t,
  const size_t,
  const bool,
  cudaStream_t);
template void thrust_local_sort<legate_type_of<LegateTypeCode::UINT32_LT>>(
  const legate_type_of<LegateTypeCode::UINT32_LT>*,
  legate_type_of<LegateTypeCode::UINT32_LT>*,
  const int64_t*,
  int64_t*,
  const size_t,
  const size_t,
  const bool,
  cudaStream_t);
template void thrust_local_sort<legate_type_of<LegateTypeCode::UINT64_LT>>(
  const legate_type_of<LegateTypeCode::UINT64_LT>*,
  legate_type_of<LegateTypeCode::UINT64_LT>*,
  const int64_t*,
  int64_t*,
  const size_t,
  const size_t,
  const bool,
  cudaStream_t);
template void thrust_local_sort<legate_type_of<LegateTypeCode::HALF_LT>>(
  const legate_type_of<LegateTypeCode::HALF_LT>*,
  legate_type_of<LegateTypeCode::HALF_LT>*,
  const int64_t*,
  int64_t*,
  const size_t,
  const size_t,
  const bool,
  cudaStream_t);
template void thrust_local_sort<legate_type_of<LegateTypeCode::FLOAT_LT>>(
  const legate_type_of<LegateTypeCode::FLOAT_LT>*,
  legate_type_of<LegateTypeCode::FLOAT_LT>*,
  const int64_t*,
  int64_t*,
  const size_t,
  const size_t,
  const bool,
  cudaStream_t);
template void thrust_local_sort<legate_type_of<LegateTypeCode::DOUBLE_LT>>(
  const legate_type_of<LegateTypeCode::DOUBLE_LT>*,
  legate_type_of<LegateTypeCode::DOUBLE_LT>*,
  const int64_t*,
  int64_t*,
  const size_t,
  const size_t,
  const bool,
  cudaStream_t);
template void thrust_local_sort<legate_type_of<LegateTypeCode::COMPLEX64_LT>>(
  const legate_type_of<LegateTypeCode::COMPLEX64_LT>*,
  legate_type_of<LegateTypeCode::COMPLEX64_LT>*,
  const int64_t*,
  int64_t*,
  const size_t,
  const size_t,
  const bool,
  cudaStream_t);
template void thrust_local_sort<legate_type_of<LegateTypeCode::COMPLEX128_LT>>(
  const legate_type_of<LegateTypeCode::COMPLEX128_LT>*,
  legate_type_of<LegateTypeCode::COMPLEX128_LT>*,
  const int64_t*,
  int64_t*,
  const size_t,
  const size_t,
  const bool,
  cudaStream_t);

template <class VAL>
void thrust_local_sort(const VAL* values_in,
                       VAL* values_out,
                       const int64_t* indices_in,
                       int64_t* indices_out,
                       const size_t volume,
                       const size_t sort_dim_size,
                       const bool stable_argsort,
                       cudaStream_t stream)
{
  if (values_in != values_out) {
    // not in-place --> need a copy
    CHECK_CUDA(cudaMemcpyAsync(
      values_out, values_in, sizeof(VAL) * volume, cudaMemcpyDeviceToDevice, stream));
  }
  if (indices_in != indices_out) {
    // not in-place --> need a copy
    CHECK_CUDA(cudaMemcpyAsync(
      indices_out, values_in, sizeof(int64_t) * volume, cudaMemcpyDeviceToDevice, stream));
  }

  if (indices_out == nullptr) {
    if (volume == sort_dim_size) {
      thrust::sort(thrust::cuda::par.on(stream), values_out, values_out + volume);
    } else {
      auto sort_id = create_buffer<uint64_t>(volume, Legion::Memory::Kind::GPU_FB_MEM);
      // init combined keys
      thrust::transform(thrust::cuda::par.on(stream),
                        thrust::make_counting_iterator<uint64_t>(0),
                        thrust::make_counting_iterator<uint64_t>(volume),
                        thrust::make_constant_iterator<uint64_t>(sort_dim_size),
                        sort_id.ptr(0),
                        thrust::divides<uint64_t>());
      auto combined = thrust::make_zip_iterator(thrust::make_tuple(sort_id.ptr(0), values_out));

      thrust::sort(thrust::cuda::par.on(stream),
                   combined,
                   combined + volume,
                   thrust::less<thrust::tuple<size_t, VAL>>());

      sort_id.destroy();
    }
  } else {
    if (volume == sort_dim_size) {
      if (stable_argsort) {
        thrust::stable_sort_by_key(
          thrust::cuda::par.on(stream), values_out, values_out + volume, indices_out);
      } else {
        thrust::sort_by_key(
          thrust::cuda::par.on(stream), values_out, values_out + volume, indices_out);
      }
    } else {
      auto sort_id = create_buffer<uint64_t>(volume, Legion::Memory::Kind::GPU_FB_MEM);
      // init combined keys
      thrust::transform(thrust::cuda::par.on(stream),
                        thrust::make_counting_iterator<uint64_t>(0),
                        thrust::make_counting_iterator<uint64_t>(volume),
                        thrust::make_constant_iterator<uint64_t>(sort_dim_size),
                        sort_id.ptr(0),
                        thrust::divides<uint64_t>());
      auto combined = thrust::make_zip_iterator(thrust::make_tuple(sort_id.ptr(0), values_out));

      if (stable_argsort) {
        thrust::stable_sort_by_key(thrust::cuda::par.on(stream),
                                   combined,
                                   combined + volume,
                                   indices_out,
                                   thrust::less<thrust::tuple<size_t, VAL>>());
      } else {
        thrust::sort_by_key(thrust::cuda::par.on(stream),
                            combined,
                            combined + volume,
                            indices_out,
                            thrust::less<thrust::tuple<size_t, VAL>>());
      }

      sort_id.destroy();
    }
  }
}

}  // namespace cunumeric