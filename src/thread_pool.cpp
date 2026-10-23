#include "../include/thread_pool.h"

#include <algorithm>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace LLaMA_Tensor{
static int g_thread_count = 0;

int GetThreadCount() {
    if (g_thread_count > 0) {
        return g_thread_count;
    }
    unsigned int hw = std::thread::hardware_concurrency();
    return hw > 0 ? static_cast<int>(hw) : 4;
}

void SetThreadCount(int n) { g_thread_count = n > 0 ? n : 0; }

void ParallelFor(int n, const std::function<void(int begin, int end)>& fn){
    if(n <= 0){
        return;
    }

    int n_threads = GetThreadCount();

    if(n < n_threads * kParallelForMinChunk){
        fn(0, n);
        return;
    }

    if(n_threads > n){
        n_threads = n;
    }

    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    std::exception_ptr first_expection = nullptr;
    std::mutex expection_lock;

    int chunk = n / n_threads;
    int remainder = n % n_threads;
    int start = 0;
    for(int i = 0;i < n_threads;i++){
        int count = chunk + (i < remainder ? 1 : 0);
        int end = start + count;
        if(count > 0){
            threads.emplace_back(
                [start, end, &fn, &first_expection, &expection_lock](){
                    try{
                        fn(start, end);
                    }catch(...){
                        std::lock_guard<std::mutex> lock(expection_lock);
                        if(!first_expection){
                            first_expection = std::current_exception();
                        }
                    }
                }
            );
            start = end;
        }
    }
    for(auto& t : threads){
        t.join();
    }
    if(first_expection){
        std::rethrow_exception(first_expection);
    }
    return;
}

}