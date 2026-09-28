#include "device_comm/device_collective/algorithms/oneshot/one_shot_types.cuh"

#include <cstdint>

#include <cooperative_groups.h>

#include "device_comm/device_collective/device_collective_kernel.cuh"
#include "device_comm/device_collective/protocols/ll/ll_primitives.cuh"
#include "device_comm/device_collective/protocols/ll/ll_value_pack.cuh"
#include "device_comm/device_primitives/value_primitives.cuh"
#include "device_comm/device_transfer/transfer_lane.cuh"
#include "pg_assert.h"

namespace mooncake {
namespace {

inline constexpr int kOneShotThreads = 512;

// Send to every peer and finish all receives before advancing LL progress.
template <typename T>
class OneShotExchange {
    using Pack = LLValuePack<T>;

   public:
    __device__ __forceinline__ OneShotExchange(const OneShotAllReducePlan& plan,
                                               LLState& ll,
                                               uint64_t timeout_ticks)
        : plan_(plan), ll_(ll), timeout_ticks_(timeout_ticks) {}

    [[nodiscard]] __device__ __forceinline__ uint64_t
    chunkElementCapacity() const {
        return OneShotBufferLayout::kChunkBytes / sizeof(T);
    }

    [[nodiscard]] __device__ __forceinline__ static uint64_t packCount(
        uint64_t count) {
        return count / Pack::kValueCount + (count % Pack::kValueCount != 0);
    }

    [[nodiscard]] __device__ __forceinline__ CollectiveStepResult
    begin(cooperative_groups::thread_block block) {
        const auto result =
            LLControl(ll_, timeout_ticks_)
                .begin(plan_.view_epoch, plan_.remotePeers(), plan_.packets,
                       plan_.layout.packetBytes(), block);
        sequence_ = ll_.progress.next_sequence;
        return result;
    }

    // Before X sends n+2, it receives Y's n+1, so Y has consumed X's n.
    // Empty/local-only calls consume no sequence, preserving this dependency.
    __device__ __forceinline__ void sendAll(
        CollectiveChunk<T> chunk,
        cooperative_groups::thread_block block) const {
        const uint64_t pack_count = packCount(chunk.count);
        const auto remote_peers = plan_.remotePeers();
        const uint64_t work = pack_count * remote_peers.size();
        const LLPacketOps packets(sequence_, timeout_ticks_);
        for (uint64_t index = block.thread_rank(); index < work;
             index += block.size()) {
            const auto& peer = remote_peers.atIndex(index / pack_count);
            const uint64_t pack_index = index % pack_count;
            const auto value =
                Pack::load(chunk.source, chunk.count, pack_index);
            const uint64_t remote_offset =
                peer.workspace_offset +
                plan_.layout.packetIndex(
                    sequence_ % OneShotBufferLayout::kSlots, plan_.self_rank,
                    pack_index * Pack::kPayloadWords) *
                    LLPacket::kStorageBytes;
            auto* destination =
                static_cast<LLPacket*>(ll_.bindings.transfer_handle->remotePtr(
                    peer.global_rank, remote_offset));
            packets.storePack<Pack>(destination, value);
        }
        block.sync();
    }

    [[nodiscard]] __device__ __forceinline__ bool readReceived(
        uint32_t active_index, uint64_t pack_index,
        typename Pack::Value& value) const {
        const auto& peer = plan_.remoteParticipant(active_index);
        const auto* packet =
            plan_.packets +
            plan_.layout.packetIndex(sequence_ % OneShotBufferLayout::kSlots,
                                     peer.in_group_rank,
                                     pack_index * Pack::kPayloadWords);
        return LLPacketOps(sequence_, timeout_ticks_)
            .loadPack<Pack>(packet, value);
    }

    [[nodiscard]] __device__ __forceinline__ CollectiveStepResult
    finish(InGroupRank failed, cooperative_groups::thread_block block) {
        return LLControl(ll_, timeout_ticks_).finish(sequence_, failed, block);
    }

   private:
    const OneShotAllReducePlan& plan_;
    LLState& ll_;
    uint64_t timeout_ticks_;
    uint64_t sequence_ = 0;
};

template <typename T, ReduceOp Op>
[[nodiscard]] __device__ __forceinline__ InGroupRank
reducePeersScalar(CollectiveChunk<T> chunk, const OneShotAllReducePlan& plan,
                  const OneShotExchange<T>& exchange, uint64_t pack_count,
                  cooperative_groups::thread_block block) {
    using Pack = LLValuePack<T>;
    for (uint64_t pack_index = block.thread_rank(); pack_index < pack_count;
         pack_index += block.size()) {
        typename Pack::Value reduced = 0;
        for (uint32_t peer = 0; peer < plan.participant_count; ++peer) {
            typename Pack::Value value;
            if (peer == plan.self_active_index) {
                value = Pack::load(chunk.source, chunk.count, pack_index);
            } else if (!exchange.readReceived(peer, pack_index, value)) {
                return plan.remoteParticipant(peer).in_group_rank;
            }
            // Start with the first participant; no identity value is needed.
            reduced =
                peer == 0 ? value : Pack::template reduce<Op>(reduced, value);
        }
        Pack::store(chunk.destination, chunk.count, pack_index, reduced);
    }
    return kInvalidInGroupRank;
}

template <typename T, ReduceOp Op>
[[nodiscard]] __device__ __forceinline__ InGroupRank
reducePeersTiled(CollectiveChunk<T> chunk, const OneShotAllReducePlan& plan,
                 const OneShotExchange<T>& exchange, uint64_t pack_count,
                 cooperative_groups::thread_block block) {
    using Pack = LLValuePack<T>;
    const auto tile = cooperative_groups::tiled_partition<8>(block);
    for (uint64_t pack_index = tile.meta_group_rank(); pack_index < pack_count;
         pack_index += tile.meta_group_size()) {
        typename Pack::Value reduced = 0;
        InGroupRank failed_rank = kInvalidInGroupRank;
        for (uint32_t base = 0; base < plan.participant_count; base += 8) {
            const uint32_t peer = base + tile.thread_rank();
            typename Pack::Value value = 0;
            InGroupRank failed = kInvalidInGroupRank;
            if (peer < plan.participant_count) {
                if (peer == plan.self_active_index) {
                    value = Pack::load(chunk.source, chunk.count, pack_index);
                } else if (!exchange.readReceived(peer, pack_index, value)) {
                    failed = plan.remoteParticipant(peer).in_group_rank;
                }
            }
#pragma unroll
            for (uint32_t lane = 0; lane < 8; ++lane) {
                const uint32_t low =
                    tile.shfl(static_cast<uint32_t>(value), lane);
                typename Pack::Value received = low;
                if constexpr (Pack::kPayloadWords == 2) {
                    const uint32_t high =
                        tile.shfl(static_cast<uint32_t>(value >> 32), lane);
                    received |= static_cast<typename Pack::Value>(high) << 32;
                }
                const auto peer_failed = tile.shfl(failed, lane);
                if (peer_failed != kInvalidInGroupRank)
                    failed_rank = peer_failed;
                if (tile.thread_rank() == 0 &&
                    base + lane < plan.participant_count)
                    reduced = base + lane == 0 ? received
                                               : Pack::template reduce<Op>(
                                                     reduced, received);
            }
            if (failed_rank != kInvalidInGroupRank) return failed_rank;
        }
        if (tile.thread_rank() == 0) {
            Pack::store(chunk.destination, chunk.count, pack_index, reduced);
        }
    }
    return kInvalidInGroupRank;
}

template <typename T, ReduceOp Op>
[[nodiscard]] __device__ __forceinline__ CollectiveStepResult runOneShotChunks(
    const AllReduceRequest& request, const OneShotAllReducePlan& plan,
    OneShotExchange<T>& exchange, cooperative_groups::thread_block block) {
    uint64_t offset = 0;
    while (offset < request.count) {
        const auto ready = exchange.begin(block);
        if (!ready.succeeded()) return ready;
        const uint64_t remaining = request.count - offset;
        const CollectiveChunk<T> chunk{
            .source = static_cast<const T*>(request.send_buffer) + offset,
            .destination = static_cast<T*>(request.recv_buffer) + offset,
            .count = remaining < exchange.chunkElementCapacity()
                         ? remaining
                         : exchange.chunkElementCapacity(),
        };
        exchange.sendAll(chunk, block);
        const uint64_t pack_count = OneShotExchange<T>::packCount(chunk.count);
        const auto failed = pack_count <= 128
                                ? reducePeersTiled<T, Op>(chunk, plan, exchange,
                                                          pack_count, block)
                                : reducePeersScalar<T, Op>(
                                      chunk, plan, exchange, pack_count, block);
        const auto result = exchange.finish(failed, block);
        if (!result.succeeded()) return result;
        offset += chunk.count;
    }
    return {};
}

template <typename T, ReduceOp Op>
[[nodiscard]] __device__ __forceinline__ CollectiveStepResult
runOneShot(const AllReduceRequest& request, OneShotAllReduceDeviceState* state,
           cooperative_groups::thread_block block) {
    const auto& plan = state->plan;
    PG_ASSERT(plan.participant_count > 0 &&
              plan.participant_count <= kMaxNumRanks);
    if (request.count == 0) return {};
    if (plan.participant_count == 1) {
        if (request.send_buffer != request.recv_buffer) {
            copyValuesTo(static_cast<const T*>(request.send_buffer),
                         request.count, block,
                         static_cast<T*>(request.recv_buffer));
        }
        return {};
    }

    PG_ASSERT(state->ll);
    OneShotExchange<T> exchange(plan, *state->ll,
                                state->collective.timeout_ticks);
    return runOneShotChunks<T, Op>(request, plan, exchange, block);
}

template <typename T, ReduceOp Op>
__global__ __launch_bounds__(kOneShotThreads) void oneShotAllReduceKernel(
    AllReduceRequest request, OneShotAllReduceDeviceState* state) {
    const auto block = cooperative_groups::this_thread_block();
    PG_ASSERT(state);
    auto result = beginCollectiveInvocation<CollectiveExecution::SingleCTA>(
        &state->plan, state->collective, block);
    if (result.succeeded()) result = runOneShot<T, Op>(request, state, block);
    finishCollectiveInvocation<CollectiveExecution::SingleCTA>(
        state->collective, state->plan.remotePeers(), block, result.failed_rank,
        request.failed_ranks_hint);
}

template <typename T, ReduceOp Op>
cudaError_t launchKernel(const AllReduceRequest& request,
                         OneShotAllReduceDeviceState* state, int threads,
                         cudaStream_t stream) {
    oneShotAllReduceKernel<T, Op><<<1, threads, 0, stream>>>(request, state);
    return cudaGetLastError();
}

template <typename T>
cudaError_t launchForType(const AllReduceRequest& request,
                          OneShotAllReduceDeviceState* state, int threads,
                          cudaStream_t stream) {
    switch (request.op) {
        case ReduceOp::Sum:
            return launchKernel<T, ReduceOp::Sum>(request, state, threads,
                                                  stream);
        case ReduceOp::Product:
            return launchKernel<T, ReduceOp::Product>(request, state, threads,
                                                      stream);
        case ReduceOp::Min:
            return launchKernel<T, ReduceOp::Min>(request, state, threads,
                                                  stream);
        case ReduceOp::Max:
            return launchKernel<T, ReduceOp::Max>(request, state, threads,
                                                  stream);
        default:
            return cudaErrorInvalidValue;
    }
}

}  // namespace

cudaError_t launchOneShotAllReduceKernel(const AllReduceRequest& request,
                                         OneShotAllReduceDeviceState* state,
                                         cudaStream_t stream) {
    if (!isDeviceAllReduceCombinationSupported(request.datatype, request.op))
        return cudaErrorInvalidValue;
    const uint64_t bytes = request.count * elementSize(request.datatype);
    // Size is fixed in a captured call; smaller blocks suit tiny messages.
    const int threads = bytes <= 512    ? 128
                        : bytes <= 1024 ? 256
                                        : kOneShotThreads;
    switch (request.datatype) {
        case DataType::Float16:
            return launchForType<__half>(request, state, threads, stream);
        case DataType::Uint8:
            return launchForType<uint8_t>(request, state, threads, stream);
        case DataType::Int8:
            return launchForType<int8_t>(request, state, threads, stream);
        case DataType::Int16:
            return launchForType<int16_t>(request, state, threads, stream);
        case DataType::Int64:
            return launchForType<int64_t>(request, state, threads, stream);
        case DataType::Bfloat16:
            return launchForType<__nv_bfloat16>(request, state, threads,
                                                stream);
        case DataType::Float32:
            return launchForType<float>(request, state, threads, stream);
        case DataType::Float64:
            return launchForType<double>(request, state, threads, stream);
        case DataType::Int32:
            return launchForType<int32_t>(request, state, threads, stream);
        case DataType::Bool:
            return launchForType<bool>(request, state, threads, stream);
        default:
            return cudaErrorInvalidValue;
    }
}

}  // namespace mooncake
