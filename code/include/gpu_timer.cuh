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

  void start() { cudaEventRecord(start_event, 0); }

  void stop() { cudaEventRecord(stop_event, 0); }

  float elapsed() {
    float elapsed;
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&elapsed, start_event, stop_event);
    return elapsed;
  }
};

#endif