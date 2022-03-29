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

#include "cunumeric/sort/sort.h"
#include "cunumeric/sort/sort_template.inl"
#include "cunumeric/sort/utilities.h"

#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/binary_search.h>
#include <thrust/tuple.h>
#include <thrust/execution_policy.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cub/thread/thread_search.cuh>

#include "cunumeric/cuda_help.h"

namespace cunumeric {

using namespace Legion;

template <LegateTypeCode CODE>
struct support_cub : std::true_type {
};
template <>
struct support_cub<LegateTypeCode::COMPLEX64_LT> : std::false_type {
};
template <>
struct support_cub<LegateTypeCode::COMPLEX128_LT> : std::false_type {
};

template <LegateTypeCode CODE, std::enable_if_t<support_cub<CODE>::value>* = nullptr>
void local_sort(const legate_type_of<CODE>* values_in,
                legate_type_of<CODE>* values_out,
                const int64_t* indices_in,
                int64_t* indices_out,
                const size_t volume,
                const size_t sort_dim_size,
                const bool stable_argsort,  // cub sort is always stable
                cudaStream_t stream)
{
  using VAL = legate_type_of<CODE>;
  // fallback to thrust approach as segmented radix sort is not suited for small segments
  if (volume == sort_dim_size || sort_dim_size > 300) {
    cub_local_sort<VAL>(
      values_in, values_out, indices_in, indices_out, volume, sort_dim_size, stream);
  } else {
    thrust_local_sort<VAL>(values_in,
                           values_out,
                           indices_in,
                           indices_out,
                           volume,
                           sort_dim_size,
                           stable_argsort,
                           stream);
  }
}

template <LegateTypeCode CODE, std::enable_if_t<!support_cub<CODE>::value>* = nullptr>
void local_sort(const legate_type_of<CODE>* values_in,
                legate_type_of<CODE>* values_out,
                const int64_t* indices_in,
                int64_t* indices_out,
                const size_t volume,
                const size_t sort_dim_size,
                const bool stable_argsort,
                cudaStream_t stream)
{
  using VAL = legate_type_of<CODE>;
  thrust_local_sort<VAL>(
    values_in, values_out, indices_in, indices_out, volume, sort_dim_size, stable_argsort, stream);
}

template <LegateTypeCode CODE, int32_t DIM>
struct SortImplBody<VariantKind::GPU, CODE, DIM> {
  using VAL = legate_type_of<CODE>;

  void operator()(const Array& input_array,
                  Array& output_array,
                  const Pitches<DIM - 1>& pitches,
                  const Rect<DIM>& rect,
                  const size_t volume,
                  const size_t sort_dim_size,
                  const bool argsort,
                  const bool stable,
                  const bool is_index_space,
                  const size_t local_rank,
                  const size_t num_ranks,
                  const std::vector<comm::Communicator>& comms)
  {
    auto input = input_array.read_accessor<VAL, DIM>(rect);

    // we allow empty domains for distributed sorting
    assert(rect.empty() || input.accessor.is_dense_row_major(rect));

    auto stream = get_cached_stream();

    // initialize sort pointers
    SortPiece<VAL> local_sorted;
    int64_t* indices_ptr = nullptr;
    VAL* values_ptr      = nullptr;
    if (argsort) {
      // make a buffer for input
      auto input_copy     = create_buffer<VAL>(volume, Legion::Memory::Kind::GPU_FB_MEM);
      local_sorted.values = input_copy;
      values_ptr          = input_copy.ptr(0);

      // initialize indices
      if (output_array.dim() == -1) {
        auto indices_buffer  = create_buffer<int64_t>(volume, Legion::Memory::Kind::GPU_FB_MEM);
        indices_ptr          = indices_buffer.ptr(0);
        local_sorted.indices = indices_buffer;
        local_sorted.size    = volume;
      } else {
        AccessorWO<int64_t, DIM> output = output_array.write_accessor<int64_t, DIM>(rect);
        assert(output.accessor.is_dense_row_major(rect));
        indices_ptr = output.ptr(rect.lo);
      }
      if (DIM == 1) {
        size_t offset = DIM > 1 ? 0 : rect.lo[0];
        if (volume > 0) {
          thrust::sequence(thrust::cuda::par.on(stream), indices_ptr, indices_ptr + volume, offset);
        }
      } else {
        thrust::transform(thrust::cuda::par.on(stream),
                          thrust::make_counting_iterator<int64_t>(0),
                          thrust::make_counting_iterator<int64_t>(volume),
                          thrust::make_constant_iterator<int64_t>(sort_dim_size),
                          indices_ptr,
                          thrust::modulus<int64_t>());
      }
    } else {
      // initialize output
      if (output_array.dim() == -1) {
        auto input_copy      = create_buffer<VAL>(volume, Legion::Memory::Kind::GPU_FB_MEM);
        values_ptr           = input_copy.ptr(0);
        local_sorted.values  = input_copy;
        local_sorted.indices = create_buffer<int64_t>(0, Legion::Memory::Kind::GPU_FB_MEM);
        local_sorted.size    = volume;
      } else {
        AccessorWO<VAL, DIM> output = output_array.write_accessor<VAL, DIM>(rect);
        assert(output.accessor.is_dense_row_major(rect));
        values_ptr = output.ptr(rect.lo);
      }
    }
    if (volume > 0) {
      // sort data (locally)
      local_sort<CODE>(input.ptr(rect.lo),
                       values_ptr,
                       indices_ptr,
                       indices_ptr,
                       volume,
                       sort_dim_size,
                       stable,
                       stream);
    }

    // this is linked to the decision in sorting.py on when to use an 'unbounded' output array.
    if (output_array.dim() == -1) {
      SortPiece<VAL> local_sorted_repartitioned =
        is_index_space
          ? sample_sort_nccl(
              local_sorted, local_rank, num_ranks, argsort, stream, comms[0].get<ncclComm_t*>())
          : local_sorted;
      if (argsort) {
        output_array.return_data(local_sorted_repartitioned.indices,
                                 local_sorted_repartitioned.size);
      } else {
        output_array.return_data(local_sorted_repartitioned.values,
                                 local_sorted_repartitioned.size);
      }
    } else if (argsort) {
      // cleanup
      local_sorted.values.destroy();
    }
  }
};

/*static*/ void SortTask::gpu_variant(TaskContext& context)
{
  sort_template<VariantKind::GPU>(context);
}

}  // namespace cunumeric
