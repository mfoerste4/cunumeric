/* Copyright 2021 NVIDIA Corporation
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

#include "unary/unary_op.h"
#include "core.h"
#include "deserializer.h"
#include "dispatch.h"
#include "point_task.h"

namespace legate {
namespace numpy {

using namespace Legion;

namespace gpu {

template <typename Function, typename ARG, typename RES>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  dense_kernel(size_t volume, Function func, RES *out, const ARG *in)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  out[idx] = func(in[idx]);
}

template <typename Function, typename ReadAcc, typename WriteAcc, typename Pitches, typename Rect>
static __global__ void __launch_bounds__(THREADS_PER_BLOCK, MIN_CTAS_PER_SM)
  generic_kernel(size_t volume, Function func, WriteAcc out, ReadAcc in, Pitches pitches, Rect rect)
{
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  auto point = pitches.unflatten(idx, rect.lo);
  out[point] = func(in[point]);
}

template <UnaryOpCode OP_CODE>
struct UnaryOpImpl {
  template <LegateTypeCode CODE,
            int DIM,
            std::enable_if_t<UnaryOp<OP_CODE, CODE>::valid> * = nullptr>
  void operator()(Shape &shape, RegionField &out_rf, RegionField &in_rf)
  {
    using OP  = UnaryOp<OP_CODE, CODE>;
    using ARG = legate_type_of<CODE>;
    using RES = std::result_of_t<OP(ARG)>;

    auto rect = shape.to_rect<DIM>();

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    if (volume == 0) return;

    auto out = out_rf.write_accessor<RES, DIM>();
    auto in  = in_rf.read_accessor<ARG, DIM>();

#ifndef LEGION_BOUNDS_CHECKS
    // Check to see if this is dense or not
    bool dense = out.accessor.is_dense_row_major(rect) && in.accessor.is_dense_row_major(rect);
#else
    // No dense execution if we're doing bounds checks
    bool dense = false;
#endif

    OP func{};
    const size_t blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (dense) {
      auto outptr = out.ptr(rect);
      auto inptr  = in.ptr(rect);
      dense_kernel<<<blocks, THREADS_PER_BLOCK>>>(volume, func, outptr, inptr);
    } else {
      generic_kernel<<<blocks, THREADS_PER_BLOCK>>>(volume, func, out, in, pitches, rect);
    }
  }

  template <LegateTypeCode CODE,
            int DIM,
            std::enable_if_t<!UnaryOp<OP_CODE, CODE>::valid> * = nullptr>
  void operator()(Shape &shape, RegionField &out_rf, RegionField &in_rf)
  {
    assert(false);
  }
};

struct UnaryOpDispatch {
  template <UnaryOpCode OP_CODE>
  void operator()(Shape &shape, RegionField &out, RegionField &in)
  {
    double_dispatch(in.dim(), in.code(), UnaryOpImpl<OP_CODE>{}, shape, out, in);
  }
};

}  // namespace gpu

/*static*/ void UnaryOpTask::gpu_variant(const Task *task,
                                         const std::vector<PhysicalRegion> &regions,
                                         Context context,
                                         Runtime *runtime)
{
  Deserializer ctx(task, regions);

  UnaryOpCode op_code;
  Shape shape;
  RegionField out;
  RegionField in;

  deserialize(ctx, op_code);
  deserialize(ctx, shape);
  deserialize(ctx, out);
  deserialize(ctx, in);

  op_dispatch(op_code, gpu::UnaryOpDispatch{}, shape, out, in);
}

}  // namespace numpy
}  // namespace legate