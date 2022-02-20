//
//  Parallel.cpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#include "Parallel.hpp"

#include "ThreadPool.hpp"

#include <cassert>
#include <sstream>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace otter {

const char* get_env_var(const char* var_name, const char* def_value = nullptr) {
    const char* value = std::getenv(var_name);
    return value ? value : def_value;
}

size_t get_env_num_threads(const char* var_name, size_t def_value = 0) {
    try {
        if (auto* value = std::getenv(var_name)) {
            int nthreads = std::stoi(value);
            assert(nthreads > 0);
            return nthreads;
        }
    } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << "Invalid " << var_name << " variable value, " << e.what();
    }
    return def_value;
}

int intraop_default_num_threads() {
    size_t nthreads = get_env_num_threads("OMP_NUM_THREADS", 0);
    if (nthreads == 0) {
        nthreads = TaskThreadPoolBase::defaultNumThreads();
    }
    return (int)nthreads;
}

std::string get_parallel_info() {
    std::ostringstream ss;
    
    ss << "OtterParallel:\n\totter::get_num_threads() : "
    << otter::get_num_threads() << std::endl;
//    ss << "\totter::get_num_interop_threads() : "
//    << otter::get_num_interop_threads() << std::endl;
    
    ss << otter::get_openmp_version() << std::endl;
    
#ifdef _OPENMP
    ss << "\tomp_get_max_threads() : " << omp_get_max_threads() << std::endl;
#endif
    
    ss << "std::thread::hardware_concurrency() : "
    << std::thread::hardware_concurrency() << std::endl;
    
    ss << "Environment variables:" << std::endl;
    ss << "\tOMP_NUM_THREADS : "
    << get_env_var("OMP_NUM_THREADS", "[not set]") << std::endl;
    
    ss << "OTTER parallel backend: ";
#if OTTER_OPENMP
    ss << "OpenMP";
#endif

    return ss.str();
}

std::string get_openmp_version() {
    std::ostringstream ss;
#ifdef _OPENMP
    {
        ss << "OpenMP " << _OPENMP;
        
        const char* ver_str = nullptr;
        switch (_OPENMP) {
            case 200505:
                ver_str = "2.5";
                break;
            case 200805:
                ver_str = "3.0";
                break;
            case 201107:
                ver_str = "3.1";
                break;
            case 201307:
                ver_str = "4.0";
                break;
            case 201511:
                ver_str = "4.5";
                break;
            default:
                ver_str = nullptr;
                break;
        }
        if (ver_str) {
            ss << " (a.k.a. OpenMP " << ver_str << ")";
        }
    }
#else
    ss << "OpenMP not found";
#endif
    return ss.str();
}

}
