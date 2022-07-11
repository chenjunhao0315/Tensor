//
//  Benchmark.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/5/22.
//

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else // _WIN32
#include <sys/time.h>
#endif // _WIN32

#include "Benchmark.hpp"

#if OTTER_BENCHMARK
#include "ConvolutionLayer.hpp"
#include "DeconvolutionLayer.hpp"

#include <stdio.h>
#endif // OTTER_BENCHMARK

namespace otter {

double get_current_time()
{
#ifdef _WIN32
    LARGE_INTEGER freq;
    LARGE_INTEGER pc;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&pc);

    return pc.QuadPart * 1000.0 / freq.QuadPart;
#else  // _WIN32
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
#endif // _WIN32
}

#if OTTER_BENCHMARK

void benchmark(const Layer* layer, double start, double end) {
    fprintf(stderr, "%-24s %-30s %8.2lfms", layer->type().c_str(), layer->name.c_str(), end - start);
    fprintf(stderr, "    |");
    fprintf(stderr, "\n");
}

void benchmark(const Layer* layer, const Tensor& bottom_blob, Tensor& top_blob, double start, double end)
{
    fprintf(stderr, "%-24s %-30s %8.2lfms", layer->type().c_str(), layer->name.c_str(), end - start);

    char in_shape_str[64] = {'\0'};
    char out_shape_str[64] = {'\0'};

    if (bottom_blob.dim() == 1) {
        sprintf(in_shape_str, "[%3lld *%lld]", bottom_blob.size(0), bottom_blob.elempack());
    }
    if (bottom_blob.dim() == 2) {
        sprintf(in_shape_str, "[%3lld, %3lld *%lld]", bottom_blob.size(1), bottom_blob.size(0), bottom_blob.elempack());
    }
    if (bottom_blob.dim() == 3) {
        sprintf(in_shape_str, "[%3lld, %3lld *%lld]", bottom_blob.size(2), bottom_blob.size(1), bottom_blob.elempack());
    }
    if (bottom_blob.dim() == 4) {
        sprintf(in_shape_str, "[%3lld, %3lld, %3lld *%lld]", bottom_blob.size(3), bottom_blob.size(2), bottom_blob.size(1), bottom_blob.elempack());
    }

    if (top_blob.dim() == 1) {
        sprintf(out_shape_str, "[%3lld *%lld]", top_blob.size(0), top_blob.elempack());
    }
    if (top_blob.dim() == 2) {
        sprintf(out_shape_str, "[%3lld, %3lld *%lld]", top_blob.size(1), top_blob.size(0), top_blob.elempack());
    }
    if (top_blob.dim() == 3) {
        sprintf(out_shape_str, "[%3lld, %3lld *%lld]", top_blob.size(2), top_blob.size(1), top_blob.elempack());
    }
    if (top_blob.dim() == 4) {
        sprintf(out_shape_str, "[%3lld, %3lld, %3lld *%lld]", top_blob.size(3), top_blob.size(2), top_blob.size(1), top_blob.elempack());
    }

    fprintf(stderr, "    | %22s -> %-22s", in_shape_str, out_shape_str);

    if (layer->type() == "Convolution") {
        fprintf(stderr, "     kernel: %1d x %1d     stride: %1d x %1d     groups: %d",
                ((ConvolutionLayer*)layer)->kernel_width,
                ((ConvolutionLayer*)layer)->kernel_height,
                ((ConvolutionLayer*)layer)->stride_width,
                ((ConvolutionLayer*)layer)->stride_height,
                ((ConvolutionLayer*)layer)->groups);
    } else if (layer->type() == "Deconvolution") {
        fprintf(stderr, "     kernel: %1d x %1d     stride: %1d x %1d     groups: %d",
                ((DeconvolutionLayer*)layer)->kernel_width,
                ((DeconvolutionLayer*)layer)->kernel_height,
                ((DeconvolutionLayer*)layer)->stride_width,
                ((DeconvolutionLayer*)layer)->stride_height,
                ((DeconvolutionLayer*)layer)->groups);
    }
    fprintf(stderr, "\n");
}

#endif

}   // end namespace otter
