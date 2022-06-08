#include "OTensor.hpp"
#include "Benchmark.hpp"

#ifdef _WIN32
#include <algorithm>
#include <windows.h> // Sleep()
#else
#include <unistd.h> // sleep()
#endif

#include <float.h>
#include <stdio.h>
#include <string.h>

static int g_warmup_loop_count = 8;
static int g_loop_count = 4;
static bool g_enable_cooling_down = true;

void benchmark(const char* model, const otter::Tensor& _in) {
    otter::Tensor in = _in;
    in.fill_(0.01f);
    
    otter::Net net;
    
#ifdef __EMSCRIPTEN__
#define MODEL_DIR "/working/"
#else
#define MODEL_DIR ""
#endif
    
    char parampath[256];
    sprintf(parampath, MODEL_DIR "%s.otter", model);
    net.load_otter(parampath, otter::CompileMode::Initial);
    
    const std::vector<const char*>& input_names = net.input_names();
    const std::vector<const char*>& output_names = net.output_names();
    
    if (g_enable_cooling_down)
    {
        // sleep 10 seconds for cooling down SOC  :(
#ifdef _WIN32
        Sleep(10 * 1000);
#elif defined(__unix__) || defined(__APPLE__)
        sleep(10);
#elif _POSIX_TIMERS
        struct timespec ts;
        ts.tv_sec = 10;
        ts.tv_nsec = 0;
        nanosleep(&ts, &ts);
#else
        // TODO How to handle it ?
#endif
    }
    
    otter::Tensor out;
    
    for (int i = 0; i < g_warmup_loop_count; i++) {
        auto ex = net.create_extractor();
        ex.input(input_names[0], in);
        ex.extract(output_names[0], out, 0);
    }

    double time_min = DBL_MAX;
    double time_max = -DBL_MAX;
    double time_avg = 0;
    
    for (int i = 0; i < g_loop_count; i++) {
        double start = otter::get_current_time();
        {
            auto ex = net.create_extractor();
            ex.input(input_names[0], in);
            ex.extract(output_names[0], out, 0);
        }
        double end = otter::get_current_time();
        double time = end - start;
        time_min = std::min(time_min, time);
        time_max = std::max(time_max, time);
        time_avg += time;
    }
    time_avg /= g_loop_count;
    fprintf(stderr, "%25s  min = %7.2f  max = %7.2f  avg = %7.2f\n", model, time_min, time_max, time_avg);
}

int main(int argc, const char * argv[]) {
    
    benchmark("nanodet-plus-m-1.5x_416_fused", otter::empty({1, 3, 416, 416}, otter::ScalarType::Float));
    
    benchmark("nanodet-plus-m-1.5x_416_int8_fused", otter::empty({1, 3, 416, 416}, otter::ScalarType::Float));
    
    benchmark("nanodet-plus-m-1.5x_416_int8_mixed", otter::empty({1, 3, 416, 416}, otter::ScalarType::Float));
    
    benchmark("simplepose_fused", otter::empty({1, 3, 256, 192}, otter::ScalarType::Float));
    
    return 0;
}

