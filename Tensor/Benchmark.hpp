//
//  Benchmark.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/5/22.
//

#ifndef Benchmark_hpp
#define Benchmark_hpp

#include "Layer.hpp"
#include "Tensor.hpp"

namespace otter {

double get_current_time();

#if OTTER_BENCHMARK

void benchmark(const Layer* layer, double start, double end);
void benchmark(const Layer* layer, const Tensor& bottom_blob, Tensor& top_blob, double start, double end);

#endif

}

#endif /* Benchmark_hpp */
