
#include <benchmark/benchmark.h>
#include <cstdint>
#include <cuda/atomic>
#include <synchronization.hpp>
#include <sys/types.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/pair.h>
#include <thrust/random.h>
#include <thrust/sequence.h>

#include "hash_functions.cuh"

auto constexpr RELAXED = cuda::std::memory_order_relaxed;
auto constexpr SEQCST = cuda::std::memory_order_seq_cst;

template <typename T> using Atomic = cuda::atomic<T, cuda::thread_scope_device>;

template <typename K, typename V>
using Slot = thrust::pair<Atomic<K>, Atomic<V>>;

template <cuda::std::memory_order K_mem_order,
          cuda::std::memory_order V_mem_order, typename K, typename V>
__global__ void find(Slot<K, V> const *slots, std::size_t num_slots, K const *k,
                     V *output, std::size_t num_keys, K empty_key,
                     V empty_value) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < num_keys) {
    auto const my_key = k[tid];
    auto const key_hash = MurmurHash3_32<K>{}(my_key);
    auto slot_index = key_hash % num_slots;
    while (true) {
      auto const current_slot = &slots[slot_index];
      auto const existing_key = current_slot->first.load(K_mem_order);

      // Matching key
      if (existing_key == my_key) {
        output[tid] = current_slot->second.load(V_mem_order);
        break;
      }
      // Empty slot. Key doesn't exist
      if (existing_key == empty_key) {
        output[tid] = empty_value;
        break;
      }
      slot_index = (slot_index + 1) % num_slots;
    }
  }
}

/**
 * @brief Generates input sizes and hash table occupancies
 *
 */
static void generate_size_and_occupancy(benchmark::internal::Benchmark *b) {
  for (auto occupancy = 40; occupancy <= 90; occupancy += 10) {
    for (auto size = 100'000'000; size <= 100'000'000; size *= 10) {
      b->Args({size, occupancy});
    }
  }
}

template <typename K, typename V, cuda::std::memory_order K_mem_order,
          cuda::std::memory_order V_mem_order>
void BM_map_proxy(benchmark::State &state) {
  auto const num_keys = state.range(0);
  auto const occupancy = (state.range(1) / double{100});
  auto const num_slots = static_cast<std::size_t>(num_keys / occupancy);
  thrust::device_vector<Slot<K, V>> slots(num_slots);

  constexpr K empty_key{-1};
  constexpr V empty_value{-1};

  // Initialize slots to empty
  thrust::for_each(thrust::device, slots.begin(), slots.end(),
                   [] __device__(auto &slot) {
                     new (&slot.first) Atomic<K>{empty_key};
                     new (&slot.second) Atomic<V>{empty_value};
                   });

  thrust::device_vector<K> keys(num_keys);
  thrust::sequence(keys.begin(), keys.end(), 0);
  auto values = thrust::make_counting_iterator(0);
  auto kvs =
      thrust::make_zip_iterator(thrust::make_tuple(keys.begin(), values));

  // Insert the key/value pairs into the map
  thrust::for_each(
      thrust::device, kvs, kvs + num_keys,
      [s = slots.data().get(),
       num_slots = slots.size()] __device__(auto const &kv) {
        auto const k = thrust::get<0>(kv);
        auto const v = thrust::get<1>(kv);
        auto const key_hash = MurmurHash3_32<K>{}(k);
        auto slot_index = key_hash % num_slots;

        while (true) {
          auto const current_slot = &s[slot_index];

          auto &slot_key = current_slot->first;
          auto &slot_value = current_slot->second;

          auto expected_key = empty_key;
          auto expected_value = empty_value;

          bool const key_success = slot_key.compare_exchange_strong(
              expected_key, k, cuda::std::memory_order_relaxed);

          bool value_success = slot_value.compare_exchange_strong(
              expected_value, v, cuda::std::memory_order_relaxed);

          // Usually, both will succeed. Otherwise, whoever won the key CAS is
          // guaranteed to eventually update the value
          if (key_success) {
            // If key succeeds and value doesn't, someone else won the value CAS
            // Spin trying to update the value
            while (not value_success) {
              value_success = slot_value.compare_exchange_strong(
                  expected_value = empty_value, v,
                  cuda::std::memory_order_relaxed);
            }
            return;
          } else if (value_success) {
            // Key CAS failed, but value succeeded. Restore the value to it's
            // initial value
            slot_value.store(empty_value, cuda::std::memory_order_relaxed);
          }

          slot_index = (slot_index + 1) % num_slots;
        }
      });

  thrust::device_vector<V> output_values(num_keys);

  for (auto _ : state) {
    cuda_event_timer raii{state};
    constexpr auto block_size{128};
    auto grid_size = (num_keys + block_size - 1) / block_size;
    find<K_mem_order, V_mem_order><<<grid_size, block_size>>>(
        slots.data().get(), num_slots, keys.data().get(),
        output_values.data().get(), num_keys, empty_key, empty_value);
  }

  state.SetBytesProcessed((sizeof(K) + sizeof(V)) *
                          int64_t(state.iterations()) *
                          int64_t(state.range(0)));
}

BENCHMARK_TEMPLATE(BM_map_proxy, int32_t, int32_t, RELAXED, RELAXED)
    ->UseManualTime()
    ->Apply(generate_size_and_occupancy)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(BM_map_proxy, int32_t, int32_t, SEQCST, RELAXED)
    ->UseManualTime()
    ->Apply(generate_size_and_occupancy)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(BM_map_proxy, int32_t, int32_t, RELAXED, SEQCST)
    ->UseManualTime()
    ->Apply(generate_size_and_occupancy)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(BM_map_proxy, int32_t, int32_t, SEQCST, SEQCST)
    ->UseManualTime()
    ->Apply(generate_size_and_occupancy)
    ->Unit(benchmark::kMillisecond);
