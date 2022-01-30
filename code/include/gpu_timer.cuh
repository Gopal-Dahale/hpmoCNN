#ifndef __GPU_TIMER_H__
#define __GPU_TIMER_H__

struct GpuTimer {
  cudaEvent_t start_event;
  cudaEvent_t stop_event;

  GpuTimer() {
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
  }

  ~GpuTimer() {
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
  }

  void start(const cudaStream_t &stream = (cudaStream_t)0) { cudaEventRecord(start_event, stream); }

  void stop(const cudaStream_t &stream = (cudaStream_t)0) { cudaEventRecord(stop_event, stream); }

  float elapsed() {
    float elapsed;
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&elapsed, start_event, stop_event);
    return elapsed;
  }
};

#endif