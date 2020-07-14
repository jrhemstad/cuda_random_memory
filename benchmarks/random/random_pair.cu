
#include <atomic>
#include <benchmark/benchmark.h>

#include <cuda/atomic>
#include <synchronization.hpp>
#include <sys/types.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/pair.h>
#include <thrust/random.h>

template <typename First, typename Second>
void BM_weak_sequential(benchmark::State &state) {

  thrust::device_vector<thrust::pair<First, Second>> input(state.range(0));
  thrust::device_vector<thrust::pair<First, Second>> output(input.size());

  for (auto _ : state) {
    cuda_event_timer raii{state};
    thrust::transform(thrust::device, input.cbegin(), input.cend(),
                      output.begin(),
                      [] __device__(auto const &p) { return p; });
  }
  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(state.range(0)) * 2 *
                          sizeof(thrust::pair<First, Second>));
}
BENCHMARK_TEMPLATE(BM_weak_sequential, int32_t, int32_t)
    ->RangeMultiplier(10)
    ->Range(100'000, 1'000'000'000)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

template <typename First, typename Second>
void BM_weak_random(benchmark::State &state) {

  thrust::device_vector<thrust::pair<First, Second>> input(state.range(0));
  thrust::device_vector<thrust::pair<First, Second>> output(input.size());

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
          return input_data[random_index];
        });
  }
  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(state.range(0)) * 2 *
                          sizeof(thrust::pair<First, Second>));
}
BENCHMARK_TEMPLATE(BM_weak_random, int32_t, int32_t)
    ->RangeMultiplier(10)
    ->Range(100'000, 1'000'000'000)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

template <typename T> using Atomic = cuda::atomic<T, cuda::thread_scope_device>;

static constexpr auto RELAXED{cuda::std::memory_order_relaxed};
static constexpr auto SEQCST{cuda::std::memory_order_seq_cst};

template <typename First, typename Second, cuda::std::memory_order F_mem_order,
          cuda::std::memory_order S_mem_order>
void BM_atomic_sequential(benchmark::State &state) {

  thrust::device_vector<thrust::pair<Atomic<First>, Atomic<Second>>> input(
      state.range(0));
  thrust::device_vector<thrust::pair<First, Second>> output(input.size());

  for (auto _ : state) {
    cuda_event_timer raii{state};
    thrust::transform(thrust::device, input.cbegin(), input.cend(),
                      output.begin(), [] __device__(auto const &p) {
                        auto const f = p.first.load(F_mem_order);
                        auto const s = p.second.load(S_mem_order);
                        return thrust::make_pair(f, s);
                      });
  }
  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(state.range(0)) * 2 *
                          sizeof(thrust::pair<First, Second>));
}
BENCHMARK_TEMPLATE(BM_atomic_sequential, int32_t, int32_t, RELAXED, RELAXED)
    ->RangeMultiplier(10)
    ->Range(100'000, 1'000'000'000)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(BM_atomic_sequential, int32_t, int32_t, SEQCST, RELAXED)
    ->RangeMultiplier(10)
    ->Range(100'000, 1'000'000'000)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(BM_atomic_sequential, int32_t, int32_t, RELAXED, SEQCST)
    ->RangeMultiplier(10)
    ->Range(100'000, 1'000'000'000)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(BM_atomic_sequential, int32_t, int32_t, SEQCST, SEQCST)
    ->RangeMultiplier(10)
    ->Range(100'000, 1'000'000'000)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

template <typename First, typename Second, cuda::std::memory_order F_mem_order,
          cuda::std::memory_order S_mem_order>
void BM_atomic_random(benchmark::State &state) {

  thrust::device_vector<thrust::pair<Atomic<First>, Atomic<Second>>> input(
      state.range(0));
  thrust::device_vector<thrust::pair<First, Second>> output(input.size());

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
          auto const &p = input_data[random_index];
          auto const f = p.first.load(F_mem_order);
          auto const s = p.second.load(S_mem_order);
          return thrust::make_pair(f, s);
        });
  }
  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(state.range(0)) * 2 *
                          sizeof(thrust::pair<First, Second>));
}
BENCHMARK_TEMPLATE(BM_atomic_random, int32_t, int32_t, RELAXED, RELAXED)
    ->RangeMultiplier(10)
    ->Range(100'000, 1'000'000'000)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_atomic_random, int32_t, int32_t, SEQCST, RELAXED)
    ->RangeMultiplier(10)
    ->Range(100'000, 1'000'000'000)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_atomic_random, int32_t, int32_t, RELAXED, SEQCST)
    ->RangeMultiplier(10)
    ->Range(100'000, 1'000'000'000)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_atomic_random, int32_t, int32_t, SEQCST, SEQCST)
    ->RangeMultiplier(10)
    ->Range(100'000, 1'000'000'000)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);

/*

template <typename T> void BM_weak_random_load(benchmark::State &state) {
thrust::device_vector<T> v(state.range(0));

auto l = [input_size = v.size()] __device__(auto i) {
thrust::default_random_engine rng;
thrust::uniform_int_distribution<int32_t> dist(0, input_size);
rng.discard(i);
return dist(rng);
};

auto const begin = thrust::make_transform_iterator(
  thrust::make_counting_iterator<int32_t>(0), l);

auto const end = thrust::make_transform_iterator(
  thrust::make_counting_iterator<int32_t>(v.size()), l);

for (auto _ : state) {
cuda_event_timer raii{state};
thrust::for_each(thrust::device, begin, end,
                 [input_data = v.data().get()] __device__(auto index) {
                   volatile auto l = input_data[index];
                 });
}
state.SetBytesProcessed(int64_t(state.iterations()) *
                      int64_t(state.range(0)) * sizeof(T));
}
BENCHMARK_TEMPLATE(BM_weak_random_load, int32_t)
->RangeMultiplier(10)
->Range(100'000, 1'000'000'000)
->UseManualTime()
->Unit(benchmark::kMillisecond);

template <typename T, cuda::std::memory_order mem_order>
void BM_atomic_sequential_load(benchmark::State &state) {
thrust::device_vector<cuda::atomic<T, cuda::thread_scope_device>> v(
  state.range(0));
for (auto _ : state) {
cuda_event_timer raii{state};
auto const begin = thrust::make_counting_iterator(0);
auto const end = begin + state.range(0);
thrust::for_each(thrust::device, begin, end,
                 [input_data = v.data().get(),
                  input_size = v.size()] __device__(auto index) {
                   volatile auto l = input_data[index].load(mem_order);
                 });
}
state.SetBytesProcessed(int64_t(state.iterations()) *
                      int64_t(state.range(0)) * sizeof(T));
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
thrust::device_vector<cuda::atomic<T, cuda::thread_scope_device>> v(
  state.range(0));

auto l = [input_size = v.size()] __device__(auto i) {
thrust::default_random_engine rng;
thrust::uniform_int_distribution<int32_t> dist(0, input_size);
rng.discard(i);
return dist(rng);
};

auto const begin = thrust::make_transform_iterator(
  thrust::make_counting_iterator<int32_t>(0), l);

auto const end = thrust::make_transform_iterator(
  thrust::make_counting_iterator<int32_t>(v.size()), l);

for (auto _ : state) {
cuda_event_timer raii{state};
thrust::for_each(thrust::device, begin, end,
                 [input_data = v.data().get()] __device__(auto index) {
                   volatile auto l = input_data[index].load(mem_order);
                 });
}
state.SetBytesProcessed(int64_t(state.iterations()) *
                      int64_t(state.range(0)) * sizeof(T));
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
*/
