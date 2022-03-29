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

#pragma once

#include "cunumeric/cunumeric.h"
#include "cunumeric/cuda_help.h"

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <typename VAL>
struct SortPiece {
  Buffer<VAL> values;
  Buffer<int64_t> indices;
  size_t size;
};

template <typename VAL>
struct Sample {
  VAL value;
  int32_t rank;
  size_t position;
};

template <class VAL>
void cub_local_sort(const VAL* values_in,
                    VAL* values_out,
                    const int64_t* indices_in,
                    int64_t* indices_out,
                    const size_t volume,
                    const size_t sort_dim_size,
                    cudaStream_t stream);

template <class VAL>
void thrust_local_sort(const VAL* values_in,
                       VAL* values_out,
                       const int64_t* indices_in,
                       int64_t* indices_out,
                       const size_t volume,
                       const size_t sort_dim_size,
                       const bool stable_argsort,
                       cudaStream_t stream);

template <typename VAL>
SortPiece<VAL> sample_sort_nccl(SortPiece<VAL> local_sorted,
                                size_t my_rank,
                                size_t num_ranks,
                                bool argsort,
                                cudaStream_t stream,
                                ncclComm_t* comm);

}  // namespace cunumeric