
#include <benchmark/benchmark.h>

#include <cuda/atomic>
#include <synchronization.hpp>
#include <sys/types.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/random.h>
#include <thrust/transform.h>

template <typename T> void BM_weak_sequential_load(benchmark::State &state) {
  thrust::device_vector<T> input(state.range(0));
  thrust::device_vector<T> output(input.size());
  for (auto _ : state) {
    cuda_event_timer raii{state};
    thrust::transform(thrust::device, input.begin(), input.end(),
                      output.begin(), [] __device__(auto v) { return v; });
  }
  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(state.range(0)) * 2 * sizeof(T));
}
BENCHMARK_TEMPLATE(BM_weak_sequential_load, int32_t)
    ->RangeMultiplier(10)
    ->Range(100'000, 1'000'000'000)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

template <typename T> void BM_weak_random_load(benchmark::State &state) {
  thrust::device_vector<T> input(state.range(0));
  thrust::device_vector<T> output(input.size());

  auto l = [input_size = input.size()] __device__(auto i) {
    thrust::default_random_engine rng;
    thrust::uniform_int_distribution<int32_t> dist(0, input_size);
    rng.discard(i);
    return dist(rng);
  };

  auto const random_begin = thrust::make_transform_iterator(
      thrust::make_counting_iterator<std::size_t>(0), l);

  auto const random_end = random_begin + input.size();

  for (auto _ : state) {
    cuda_event_timer raii{state};

    thrust::transform(
        thrust::device, random_begin, random_end, output.begin(),
        [input_data = input.data().get()] __device__(auto random_index) {
          return input_data[random_index];
        });
  }
  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(state.range(0)) * 2 * sizeof(T));
}
BENCHMARK_TEMPLATE(BM_weak_random_load, int32_t)
    ->RangeMultiplier(10)
    ->Range(100'000, 1'000'000'000)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

template <typename U> using Atomic = cuda::atomic<U, cuda::thread_scope_device>;

template <typename T, cuda::std::memory_order mem_order>
void BM_atomic_sequential_load(benchmark::State &state) {

  thrust::device_vector<Atomic<T>> input(state.range(0));
  thrust::device_vector<T> output(input.size());

  for (auto _ : state) {
    cuda_event_timer raii{state};
    thrust::transform(thrust::device, input.cbegin(), input.cend(),
                      output.begin(), [] __device__(auto const &atomic_value) {
                        return atomic_value.load(mem_order);
                      });
  }
  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(state.range(0)) * 2 * sizeof(T));
}
BENCHMARK_TEMPLATE(BM_atomic_sequential_load, int32_t,
                   cuda::std::memory_order_relaxed)
    ->RangeMultiplier(10)
    ->Range(100'000, 1'000'000'000)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(BM_atomic_sequential_load, int32_t,
                   cuda::std::memory_order_seq_cst)
    ->RangeMultiplier(10)
    ->Range(100'000, 1'000'000'000)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

template <typename T, cuda::std::memory_order mem_order>
void BM_atomic_random_load(benchmark::State &state) {
  thrust::device_vector<Atomic<T>> input(state.range(0));
  thrust::device_vector<T> output(input.size());

  auto l = [input_size = input.size()] __device__(auto i) {
    thrust::default_random_engine rng;
    thrust::uniform_int_distribution<int32_t> dist(0, input_size);
    rng.discard(i);
    return dist(rng);
  };

  auto const random_begin = thrust::make_transform_iterator(
      thrust::make_counting_iterator<int32_t>(0), l);

  auto const random_end = random_begin + input.size();

  for (auto _ : state) {
    cuda_event_timer raii{state};
    thrust::transform(
        thrust::device, random_begin, random_end, output.begin(),
        [input_data = input.data().get()] __device__(auto random_index) {
          return input_data[random_index].load(mem_order);
        });
  }
  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(state.range(0)) * 2 * sizeof(T));
}
BENCHMARK_TEMPLATE(BM_atomic_random_load, int32_t,
                   cuda::std::memory_order_relaxed)
    ->RangeMultiplier(10)
    ->Range(100'000, 1'000'000'000)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(BM_atomic_random_load, int32_t,
                   cuda::std::memory_order_seq_cst)
    ->RangeMultiplier(10)
    ->Range(100'000, 1'000'000'000)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);