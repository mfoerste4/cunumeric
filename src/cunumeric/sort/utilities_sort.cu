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
#include <cub/thread/thread_search.cuh>

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
template SortPiece<legate_type_of<LegateTypeCode::BOOL_LT>> sample_sort_nccl<
  legate_type_of<LegateTypeCode::BOOL_LT>>(SortPiece<legate_type_of<LegateTypeCode::BOOL_LT>>,
                                           size_t,
                                           size_t,
                                           bool,
                                           cudaStream_t,
                                           ncclComm_t*);
template SortPiece<legate_type_of<LegateTypeCode::INT8_LT>> sample_sort_nccl<
  legate_type_of<LegateTypeCode::INT8_LT>>(SortPiece<legate_type_of<LegateTypeCode::INT8_LT>>,
                                           size_t,
                                           size_t,
                                           bool,
                                           cudaStream_t,
                                           ncclComm_t*);
template SortPiece<legate_type_of<LegateTypeCode::INT16_LT>> sample_sort_nccl<
  legate_type_of<LegateTypeCode::INT16_LT>>(SortPiece<legate_type_of<LegateTypeCode::INT16_LT>>,
                                            size_t,
                                            size_t,
                                            bool,
                                            cudaStream_t,
                                            ncclComm_t*);
template SortPiece<legate_type_of<LegateTypeCode::INT32_LT>> sample_sort_nccl<
  legate_type_of<LegateTypeCode::INT32_LT>>(SortPiece<legate_type_of<LegateTypeCode::INT32_LT>>,
                                            size_t,
                                            size_t,
                                            bool,
                                            cudaStream_t,
                                            ncclComm_t*);
template SortPiece<legate_type_of<LegateTypeCode::INT64_LT>> sample_sort_nccl<
  legate_type_of<LegateTypeCode::INT64_LT>>(SortPiece<legate_type_of<LegateTypeCode::INT64_LT>>,
                                            size_t,
                                            size_t,
                                            bool,
                                            cudaStream_t,
                                            ncclComm_t*);
template SortPiece<legate_type_of<LegateTypeCode::UINT8_LT>> sample_sort_nccl<
  legate_type_of<LegateTypeCode::UINT8_LT>>(SortPiece<legate_type_of<LegateTypeCode::UINT8_LT>>,
                                            size_t,
                                            size_t,
                                            bool,
                                            cudaStream_t,
                                            ncclComm_t*);
template SortPiece<legate_type_of<LegateTypeCode::UINT16_LT>> sample_sort_nccl<
  legate_type_of<LegateTypeCode::UINT16_LT>>(SortPiece<legate_type_of<LegateTypeCode::UINT16_LT>>,
                                             size_t,
                                             size_t,
                                             bool,
                                             cudaStream_t,
                                             ncclComm_t*);
template SortPiece<legate_type_of<LegateTypeCode::UINT32_LT>> sample_sort_nccl<
  legate_type_of<LegateTypeCode::UINT32_LT>>(SortPiece<legate_type_of<LegateTypeCode::UINT32_LT>>,
                                             size_t,
                                             size_t,
                                             bool,
                                             cudaStream_t,
                                             ncclComm_t*);
template SortPiece<legate_type_of<LegateTypeCode::UINT64_LT>> sample_sort_nccl<
  legate_type_of<LegateTypeCode::UINT64_LT>>(SortPiece<legate_type_of<LegateTypeCode::UINT64_LT>>,
                                             size_t,
                                             size_t,
                                             bool,
                                             cudaStream_t,
                                             ncclComm_t*);
template SortPiece<legate_type_of<LegateTypeCode::HALF_LT>> sample_sort_nccl<
  legate_type_of<LegateTypeCode::HALF_LT>>(SortPiece<legate_type_of<LegateTypeCode::HALF_LT>>,
                                           size_t,
                                           size_t,
                                           bool,
                                           cudaStream_t,
                                           ncclComm_t*);
template SortPiece<legate_type_of<LegateTypeCode::FLOAT_LT>> sample_sort_nccl<
  legate_type_of<LegateTypeCode::FLOAT_LT>>(SortPiece<legate_type_of<LegateTypeCode::FLOAT_LT>>,
                                            size_t,
                                            size_t,
                                            bool,
                                            cudaStream_t,
                                            ncclComm_t*);
template SortPiece<legate_type_of<LegateTypeCode::DOUBLE_LT>> sample_sort_nccl<
  legate_type_of<LegateTypeCode::DOUBLE_LT>>(SortPiece<legate_type_of<LegateTypeCode::DOUBLE_LT>>,
                                             size_t,
                                             size_t,
                                             bool,
                                             cudaStream_t,
                                             ncclComm_t*);
template SortPiece<legate_type_of<LegateTypeCode::COMPLEX64_LT>>
sample_sort_nccl<legate_type_of<LegateTypeCode::COMPLEX64_LT>>(
  SortPiece<legate_type_of<LegateTypeCode::COMPLEX64_LT>>,
  size_t,
  size_t,
  bool,
  cudaStream_t,
  ncclComm_t*);
template SortPiece<legate_type_of<LegateTypeCode::COMPLEX128_LT>>
sample_sort_nccl<legate_type_of<LegateTypeCode::COMPLEX128_LT>>(
  SortPiece<legate_type_of<LegateTypeCode::COMPLEX128_LT>>,
  size_t,
  size_t,
  bool,
  cudaStream_t,
  ncclComm_t*);

// auto align to multiples of 16 bytes
auto get_16b_aligned = [](auto bytes) { return std::max<size_t>(16, (bytes + 15) / 16 * 16); };
auto get_16b_aligned_count = [](auto count, auto element_bytes) {
  return (get_16b_aligned(count * element_bytes) + element_bytes - 1) / element_bytes;
};

template <typename VAL>
struct SampleComparator : public thrust::binary_function<Sample<VAL>, Sample<VAL>, bool> {
  __host__ __device__ bool operator()(const Sample<VAL>& lhs, const Sample<VAL>& rhs) const
  {
    // special case for unused samples
    if (lhs.rank < 0 || rhs.rank < 0) { return rhs.rank < 0 && lhs.rank >= 0; }

    if (lhs.value != rhs.value) {
      return lhs.value < rhs.value;
    } else if (lhs.rank != rhs.rank) {
      return lhs.rank < rhs.rank;
    } else {
      return lhs.position < rhs.position;
    }
  }
};

template <typename VAL>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  extract_samples(const VAL* data,
                  const size_t volume,
                  Sample<VAL>* samples,
                  const size_t num_local_samples,
                  const Sample<VAL> init_sample,
                  const size_t offset,
                  const size_t rank)
{
  const size_t sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample_idx >= num_local_samples) return;

  if (num_local_samples < volume) {
    const size_t index                    = (sample_idx + 1) * volume / num_local_samples - 1;
    samples[offset + sample_idx].value    = data[index];
    samples[offset + sample_idx].rank     = rank;
    samples[offset + sample_idx].position = index;
  } else {
    // edge case where num_local_samples > volume
    if (sample_idx < volume) {
      samples[offset + sample_idx].value    = data[sample_idx];
      samples[offset + sample_idx].rank     = rank;
      samples[offset + sample_idx].position = sample_idx;
    } else {
      samples[offset + sample_idx] = init_sample;
    }
  }
}

template <typename VAL>
__global__ static void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  extract_split_positions(const VAL* data,
                          const size_t volume,
                          const Sample<VAL>* samples,
                          const size_t num_samples,
                          size_t* split_positions,
                          const size_t num_splitters,
                          const size_t rank)
{
  const size_t splitter_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (splitter_idx >= num_splitters) return;

  const size_t index         = (splitter_idx + 1) * num_samples / (num_splitters + 1) - 1;
  const Sample<VAL> splitter = samples[index];

  // now perform search on data to receive position *after* last element to be
  // part of the package for rank splitter_idx
  if (rank > splitter.rank) {
    // position of the last position with smaller value than splitter.value + 1
    split_positions[splitter_idx] = cub::LowerBound(data, volume, splitter.value);
  } else if (rank < splitter.rank) {
    // position of the first position with value larger than splitter.value
    split_positions[splitter_idx] = cub::UpperBound(data, volume, splitter.value);
  } else {
    split_positions[splitter_idx] = splitter.position + 1;
  }
}

template <typename VAL>
SortPiece<VAL> sample_sort_nccl(SortPiece<VAL> local_sorted,
                                size_t my_rank,
                                size_t num_ranks,
                                bool argsort,
                                cudaStream_t stream,
                                ncclComm_t* comm)
{
  size_t volume = local_sorted.size;

  // collect local samples - for now we take num_ranks samples for every node
  // worst case this leads to 2*N/ranks elements on a single node
  size_t num_local_samples = num_ranks;

  size_t num_global_samples = num_local_samples * num_ranks;
  auto samples              = create_buffer<Sample<VAL>>(num_global_samples, Memory::GPU_FB_MEM);

  Sample<VAL> init_sample;
  {
    const size_t num_blocks = (num_local_samples + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    init_sample.rank        = -1;  // init samples that are not populated
    size_t offset           = num_local_samples * my_rank;
    extract_samples<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(local_sorted.values.ptr(0),
                                                                  volume,
                                                                  samples.ptr(0),
                                                                  num_local_samples,
                                                                  init_sample,
                                                                  offset,
                                                                  my_rank);
  }

  // AllGather: check alignment? as we want to receive data in-place we take exact size for now
  CHECK_NCCL(ncclAllGather(samples.ptr(my_rank * num_ranks),
                           samples.ptr(0),
                           num_ranks * sizeof(Sample<VAL>),
                           ncclInt8,
                           *comm,
                           stream));

  // sort samples on device
  thrust::stable_sort(thrust::cuda::par.on(stream),
                      samples.ptr(0),
                      samples.ptr(0) + num_global_samples,
                      SampleComparator<VAL>());

  auto lower_bound          = thrust::lower_bound(thrust::cuda::par.on(stream),
                                         samples.ptr(0),
                                         samples.ptr(0) + num_global_samples,
                                         init_sample,
                                         SampleComparator<VAL>());
  size_t num_usable_samples = lower_bound - samples.ptr(0);

  // select splitters / positions based on samples (on device)
  const size_t num_splitters = num_ranks - 1;
  auto split_positions       = create_buffer<size_t>(num_splitters, Memory::Z_COPY_MEM);
  {
    const size_t num_blocks = (num_splitters + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    extract_split_positions<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
      local_sorted.values.ptr(0),
      volume,
      samples.ptr(0),
      num_usable_samples,
      split_positions.ptr(0),
      num_splitters,
      my_rank);
  }

  // need to sync as we share values in between host/device
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // collect sizes2send, send to rank i: local_sort_data from positions  split_positions[i-1],
  // split_positions[i] - 1
  auto size_send = create_buffer<size_t>(num_ranks, Memory::Z_COPY_MEM);
  {
    size_t last_position = 0;
    for (size_t rank = 0; rank < num_ranks - 1; ++rank) {
      size_t cur_position = split_positions[rank];
      size_send[rank]     = cur_position - last_position;
      last_position       = cur_position;
    }
    size_send[num_ranks - 1] = volume - last_position;
  }

  // cleanup intermediate data structures
  samples.destroy();
  split_positions.destroy();

  // all2all exchange send/receive sizes
  auto size_recv = create_buffer<size_t>(num_ranks, Memory::Z_COPY_MEM);
  CHECK_NCCL(ncclGroupStart());
  for (size_t r = 0; r < num_ranks; r++) {
    CHECK_NCCL(ncclSend(size_send.ptr(r), 1, ncclUint64, r, *comm, stream));
    CHECK_NCCL(ncclRecv(size_recv.ptr(r), 1, ncclUint64, r, *comm, stream));
  }
  CHECK_NCCL(ncclGroupEnd());

  // need to sync as we share values in between host/device
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // handle alignment
  std::vector<size_t> aligned_pos_vals_send(num_ranks);
  std::vector<size_t> aligned_pos_idcs_send(num_ranks);
  size_t buf_size_send_vals_total = 0;
  size_t buf_size_send_idcs_total = 0;
  for (size_t i = 0; i < num_ranks; ++i) {
    // align buffer to allow data transfer of 16byte blocks
    aligned_pos_vals_send[i] = buf_size_send_vals_total;
    buf_size_send_vals_total += get_16b_aligned_count(size_send[i], sizeof(VAL));
    if (argsort) {
      aligned_pos_idcs_send[i] = buf_size_send_idcs_total;
      buf_size_send_idcs_total += get_16b_aligned_count(size_send[i], sizeof(int64_t));
    }
  }

  // copy values into aligned send buffer
  auto val_send_buf = local_sorted.values;
  if (buf_size_send_vals_total > volume) {
    val_send_buf = create_buffer<VAL>(buf_size_send_vals_total, Memory::GPU_FB_MEM);
    size_t pos   = 0;
    for (size_t r = 0; r < num_ranks; ++r) {
      CHECK_CUDA(cudaMemcpyAsync(val_send_buf.ptr(aligned_pos_vals_send[r]),
                                 local_sorted.values.ptr(pos),
                                 sizeof(VAL) * size_send[r],
                                 cudaMemcpyDeviceToDevice,
                                 stream));
      pos += size_send[r];
    }
    local_sorted.values.destroy();
  }

  // copy indices into aligned send buffer
  auto idc_send_buf = local_sorted.indices;
  if (argsort && buf_size_send_idcs_total > volume) {
    idc_send_buf = create_buffer<int64_t>(buf_size_send_idcs_total, Memory::GPU_FB_MEM);
    size_t pos   = 0;
    for (size_t r = 0; r < num_ranks; ++r) {
      CHECK_CUDA(cudaMemcpyAsync(idc_send_buf.ptr(aligned_pos_idcs_send[r]),
                                 local_sorted.indices.ptr(pos),
                                 sizeof(int64_t) * size_send[r],
                                 cudaMemcpyDeviceToDevice,
                                 stream));
      pos += size_send[r];
    }
    local_sorted.indices.destroy();
  }

  // allocate target buffers
  std::vector<SortPiece<VAL>> merge_buffers(num_ranks);
  for (size_t i = 0; i < num_ranks; ++i) {
    auto buf_size_vals_recv = get_16b_aligned_count(size_recv[i], sizeof(VAL));
    merge_buffers[i].values = create_buffer<VAL>(buf_size_vals_recv, Memory::GPU_FB_MEM);
    merge_buffers[i].size   = size_recv[i];
    if (argsort) {
      auto buf_size_idcs_recv  = get_16b_aligned_count(size_recv[i], sizeof(int64_t));
      merge_buffers[i].indices = create_buffer<int64_t>(buf_size_idcs_recv, Memory::GPU_FB_MEM);
    } else {
      merge_buffers[i].indices = create_buffer<int64_t>(0, Memory::GPU_FB_MEM);
    }
  }
  CHECK_NCCL(ncclGroupStart());
  for (size_t r = 0; r < num_ranks; r++) {
    CHECK_NCCL(ncclSend(val_send_buf.ptr(aligned_pos_vals_send[r]),
                        get_16b_aligned(size_send[r] * sizeof(VAL)),
                        ncclInt8,
                        r,
                        *comm,
                        stream));
    CHECK_NCCL(ncclRecv(merge_buffers[r].values.ptr(0),
                        get_16b_aligned(size_recv[r] * sizeof(VAL)),
                        ncclInt8,
                        r,
                        *comm,
                        stream));
  }
  CHECK_NCCL(ncclGroupEnd());

  if (argsort) {
    CHECK_NCCL(ncclGroupStart());
    for (size_t r = 0; r < num_ranks; r++) {
      CHECK_NCCL(ncclSend(idc_send_buf.ptr(aligned_pos_idcs_send[r]),
                          get_16b_aligned_count(size_send[r], sizeof(int64_t)),
                          ncclInt64,
                          r,
                          *comm,
                          stream));
      CHECK_NCCL(ncclRecv(merge_buffers[r].indices.ptr(0),
                          get_16b_aligned_count(size_recv[r], sizeof(int64_t)),
                          ncclInt64,
                          r,
                          *comm,
                          stream));
    }
    CHECK_NCCL(ncclGroupEnd());
  }

  // cleanup remaining buffers
  size_send.destroy();
  size_recv.destroy();
  val_send_buf.destroy();
  idc_send_buf.destroy();

  // now merge sort all into the result buffer
  // maybe k-way merge is more efficient here...
  for (size_t stride = 1; stride < num_ranks; stride *= 2) {
    for (size_t pos = 0; pos + stride < num_ranks; pos += 2 * stride) {
      SortPiece<VAL> source1 = merge_buffers[pos];
      SortPiece<VAL> source2 = merge_buffers[pos + stride];
      auto merged_size       = source1.size + source2.size;
      auto merged_values     = create_buffer<VAL>(merged_size);
      auto merged_indices    = source1.indices;  // will be overriden for argsort
      auto p_merged_values   = merged_values.ptr(0);
      auto p_values1         = source1.values.ptr(0);
      auto p_values2         = source2.values.ptr(0);
      if (argsort) {
        merged_indices = create_buffer<int64_t>(merged_size);
        // merge with key/value
        auto p_indices1       = source1.indices.ptr(0);
        auto p_indices2       = source2.indices.ptr(0);
        auto p_merged_indices = merged_indices.ptr(0);
        thrust::merge_by_key(thrust::cuda::par.on(stream),
                             p_values1,
                             p_values1 + source1.size,
                             p_values2,
                             p_values2 + source2.size,
                             p_indices1,
                             p_indices2,
                             p_merged_values,
                             p_merged_indices);
        source1.indices.destroy();
      } else {
        thrust::merge(thrust::cuda::par.on(stream),
                      p_values1,
                      p_values1 + source1.size,
                      p_values2,
                      p_values2 + source2.size,
                      p_merged_values);
      }

      source1.values.destroy();
      source2.values.destroy();
      source2.indices.destroy();

      merge_buffers[pos].values  = merged_values;
      merge_buffers[pos].indices = merged_indices;
      merge_buffers[pos].size    = merged_size;
    }
  }
  return merge_buffers[0];
}

}  // namespace cunumeric