//
//  ThreadPool.hpp
//  Tensor
//
//  Created by 陳均豪 on 2022/1/29.
//

#ifndef ThreadPool_hpp
#define ThreadPool_hpp

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>
#include <thread>
#include <queue>

namespace otter {

class TaskThreadPoolBase {
public:
    virtual void run(std::function<void()> func) = 0;
    
    virtual size_t size() const = 0;
    
    // Idle threads in this pool
    virtual size_t numAvailable() const = 0;
    
    // Check the thread is come from this pool or not
    virtual bool inThreadPool() const = 0;
    
    virtual ~TaskThreadPoolBase() noexcept {}
    
    static size_t defaultNumThreads() {
        auto num_threads = std::thread::hardware_concurrency();
#if defined(_M_X64) || defined(__x86_64__)
        num_threads /= 2;
#endif
        return num_threads;
    }
};

class ThreadPool : public TaskThreadPoolBase {
protected:
    struct task_element_t {
        bool run_with_id;
        const std::function<void()> no_id;
        const std::function<void(std::size_t)> with_id;
        
        explicit task_element_t(std::function<void()> f) : run_with_id(false), no_id(std::move(f)), with_id(nullptr) {}
        explicit task_element_t(std::function<void(std::size_t)> f) : run_with_id(true), no_id(nullptr), with_id(std::move(f)) {}
    };
    
    std::queue<task_element_t> tasks_;
    std::vector<std::thread> threads_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::condition_variable completed_;
    std::atomic_bool running_;
    bool complete_;
    std::size_t available_;
    std::size_t total_;
    int numa_node_id_;
    
public:
    ThreadPool() = delete;
    
    explicit ThreadPool(int pool_size, int numa_node_id = -1, std::function<void()> init_thread = nullptr);
    
    ~ThreadPool() override;
    
    size_t size() const override;
    
    size_t numAvailable() const override;
    
    bool inThreadPool() const override;
    
    void run(std::function<void()> func) override;
    
    template <typename Task>
    void runTaskWithID(Task task) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        // Set task and signal condition variable so that a worker thread will
        // wake up and use the task.
        tasks_.emplace(static_cast<std::function<void(std::size_t)>>(task));
        complete_ = false;
        condition_.notify_one();
    }
    
    void waitWorkComplete();
    
private:
    // @brief Entry point for pool threads.
    void main_loop(std::size_t index);
};

class TaskThreadPool : public ThreadPool {
public:
    explicit TaskThreadPool(size_t pool_size, int numa_node_id = -1) : ThreadPool((int)pool_size, numa_node_id,
        [numa_node_id]() {
//          setThreadName("CaffeTaskThread");
//          NUMABind(numa_node_id);
        }) {}
};





}

#endif /* ThreadPool_hpp */
