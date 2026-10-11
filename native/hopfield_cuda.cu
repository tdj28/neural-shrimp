// Optional float64 batch trace backend. The CPU reference remains unchanged.
// Compile with --fmad=false, without fast math, for the explicitly named GPU.
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <initializer_list>

extern "C" {
struct HopfieldCudaInfo {
    char name[256];
    int device, device_count, major, minor, multiprocessors;
    int runtime_version, driver_version, single_to_double_ratio;
    std::uint64_t total_bytes, free_bytes;
};
struct HopfieldCudaTiming {
    double setup_seconds, allocation_seconds, host_to_device_seconds;
    double kernel_seconds, kernel_wall_seconds, device_to_host_seconds;
    double free_seconds, total_seconds;
    std::uint64_t requested_bytes, free_before_bytes, total_device_bytes;
    int device_output_copied;
};
}

namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now()-start).count();
}
void message(char* out, std::size_t capacity, const char* operation, cudaError_t code) {
    if (out && capacity) std::snprintf(out, capacity, "%s: %s (%d)",
                                     operation, cudaGetErrorString(code), int(code));
}
__device__ void rhs(const double* x, double* f, double alpha, double gamma) {
    const double a = tanh(x[0]), b = tanh(x[1]), c = tanh(x[2]);
    f[0] = -x[0] + 1.5*a + 2.9*b + alpha*c;
    f[1] = -x[1] - 1.5*a + 1.18*b;
    f[2] = -x[2] + gamma*a - 22.0*b + 0.47*c;
}
__device__ bool finite3(const double* x) {
    return isfinite(x[0]) && isfinite(x[1]) && isfinite(x[2]);
}
__global__ void traces(const double* alpha, const double* gamma, const double* initial,
                       int n, double dt, int steps, int stride, int samples,
                       double* states, int* statuses, double* final,
                       std::int64_t* progress) {
    const int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= n) return;
    double y[3], f[3], trial[3], k2[3], k3[3], k4[3];
    double* output = states + std::uint64_t(index)*samples*3;
    for (int j=0; j<3; ++j) output[j] = y[j] = initial[3LL*index+j];
    rhs(y, f, alpha[index], gamma[index]);
    int done=0, retained=1;
    bool failed=!finite3(f);
    for (int step=1; step<=steps && !failed; ++step) {
        // Identical stage order and endpoint-RHS reuse to native/hopfield.cpp.
        for (int j=0; j<3; ++j) trial[j] = y[j] + 0.5*dt*f[j];
        rhs(trial, k2, alpha[index], gamma[index]);
        for (int j=0; j<3; ++j) trial[j] = y[j] + 0.5*dt*k2[j];
        rhs(trial, k3, alpha[index], gamma[index]);
        for (int j=0; j<3; ++j) trial[j] = y[j] + dt*k3[j];
        rhs(trial, k4, alpha[index], gamma[index]);
        for (int j=0; j<3; ++j)
            y[j] += (dt/6.0)*(f[j] + 2.0*k2[j] + 2.0*k3[j] + k4[j]);
        rhs(y, f, alpha[index], gamma[index]);
        done=step;
        failed=!finite3(y) || !finite3(f);
        if (!failed && step%stride==0) {
            for (int j=0; j<3; ++j) output[3LL*retained+j]=y[j];
            ++retained;
        }
    }
    for (int j=0; j<3; ++j) final[3LL*index+j]=y[j];
    progress[2LL*index]=done;
    progress[2LL*index+1]=retained;
    statuses[index]=failed ? -2 : retained;
}
}

extern "C" int hopfield_cuda_device_info(int device, HopfieldCudaInfo* out,
                                          char* error, std::size_t error_capacity) {
    if (!out) return -1;
    *out={};
    cudaError_t code=cudaGetDeviceCount(&out->device_count);
    if (code!=cudaSuccess) { message(error,error_capacity,"cudaGetDeviceCount",code); return int(code); }
    if (device<0 || device>=out->device_count) {
        if (error && error_capacity) std::snprintf(error,error_capacity,"Invalid CUDA device index");
        return -1;
    }
    code=cudaSetDevice(device);
    if (code!=cudaSuccess) { message(error,error_capacity,"cudaSetDevice",code); return int(code); }
    cudaDeviceProp prop{};
    code=cudaGetDeviceProperties(&prop,device);
    if (code!=cudaSuccess) { message(error,error_capacity,"cudaGetDeviceProperties",code); return int(code); }
    std::size_t free_bytes=0,total_bytes=0;
    code=cudaMemGetInfo(&free_bytes,&total_bytes);
    if (code!=cudaSuccess) { message(error,error_capacity,"cudaMemGetInfo",code); return int(code); }
    out->device=device; out->major=prop.major; out->minor=prop.minor;
    out->multiprocessors=prop.multiProcessorCount;
    out->free_bytes=free_bytes; out->total_bytes=total_bytes;
    std::snprintf(out->name,sizeof(out->name),"%s",prop.name);
    code=cudaRuntimeGetVersion(&out->runtime_version);
    if (code==cudaSuccess) code=cudaDriverGetVersion(&out->driver_version);
    if (code==cudaSuccess) code=cudaDeviceGetAttribute(&out->single_to_double_ratio,
        cudaDevAttrSingleToDoublePrecisionPerfRatio,device);
    if (code!=cudaSuccess) { message(error,error_capacity,"CUDA version/device attribute",code); return int(code); }
    return 0;
}

extern "C" int hopfield_cuda_batch(const double* alpha, const double* gamma,
    const double* initial, int n, double dt, int steps, int stride, int samples,
    int device, std::uint64_t max_bytes, double* states, int* statuses,
    double* final, std::int64_t* progress, HopfieldCudaTiming* timing,
    char* error, std::size_t error_capacity) {
    if (!timing) return -1;
    *timing={};
    const auto total_start=Clock::now();
    if (!alpha || !gamma || !initial || !states || !statuses || !final || !progress ||
        n<=0 || !std::isfinite(dt) || dt<=0 || steps<1 || steps>1000000 || stride<1 ||
        steps%stride || samples!=steps/stride+1) {
        if (error && error_capacity) std::snprintf(error,error_capacity,"Invalid batch dimensions/grid/pointers");
        timing->total_seconds=elapsed(total_start); return -1;
    }
    const std::uint64_t n64=n, state_bytes=n64*std::uint64_t(samples)*3*sizeof(double);
    const std::uint64_t required=state_bytes+n64*(8*sizeof(double)+sizeof(int)+2*sizeof(std::int64_t));
    timing->requested_bytes=required;
    if (required>max_bytes) {
        if (error && error_capacity) std::snprintf(error,error_capacity,"Batch exceeds configured device-byte ceiling");
        timing->total_seconds=elapsed(total_start); return -1;
    }
    HopfieldCudaInfo info{};
    int result=hopfield_cuda_device_info(device,&info,error,error_capacity);
    timing->setup_seconds=elapsed(total_start);
    timing->free_before_bytes=info.free_bytes; timing->total_device_bytes=info.total_bytes;
    // Leave at least half currently free memory for shared workloads/headroom.
    if (!result && required>info.free_bytes/2) {
        if (error && error_capacity) std::snprintf(error,error_capacity,"Batch exceeds half currently free VRAM");
        result=-1;
    }
    if (result) { timing->total_seconds=elapsed(total_start); return result; }
    double *da=nullptr,*dg=nullptr,*di=nullptr,*ds=nullptr,*df=nullptr;
    int* dstatus=nullptr; std::int64_t* dp=nullptr;
    cudaEvent_t event_start=nullptr,event_end=nullptr;
    auto checked=[&](cudaError_t code,const char* name) {
        if (code!=cudaSuccess && !result) { result=int(code); message(error,error_capacity,name,code); }
        return code==cudaSuccess;
    };
    auto stage=Clock::now();
    checked(cudaMalloc((void**)&da,n64*sizeof(double)),"allocate alpha");
    if (!result) checked(cudaMalloc((void**)&dg,n64*sizeof(double)),"allocate gamma");
    if (!result) checked(cudaMalloc((void**)&di,n64*3*sizeof(double)),"allocate initial");
    if (!result) checked(cudaMalloc((void**)&ds,state_bytes),"allocate states");
    if (!result) checked(cudaMalloc((void**)&dstatus,n64*sizeof(int)),"allocate statuses");
    if (!result) checked(cudaMalloc((void**)&df,n64*3*sizeof(double)),"allocate final");
    if (!result) checked(cudaMalloc((void**)&dp,n64*2*sizeof(std::int64_t)),"allocate progress");
    if (!result) checked(cudaMemset(ds,0xff,state_bytes),"initialize unretained states to NaN");
    if (!result) checked(cudaEventCreate(&event_start),"create start event");
    if (!result) checked(cudaEventCreate(&event_end),"create end event");
    timing->allocation_seconds=elapsed(stage); stage=Clock::now();
    if (!result) checked(cudaMemcpy(da,alpha,n64*sizeof(double),cudaMemcpyHostToDevice),"copy alpha");
    if (!result) checked(cudaMemcpy(dg,gamma,n64*sizeof(double),cudaMemcpyHostToDevice),"copy gamma");
    if (!result) checked(cudaMemcpy(di,initial,n64*3*sizeof(double),cudaMemcpyHostToDevice),"copy initial");
    timing->host_to_device_seconds=elapsed(stage); stage=Clock::now();
    if (!result) checked(cudaEventRecord(event_start),"record start event");
    if (!result) {
        traces<<<(n+127LL)/128,128>>>(da,dg,di,n,dt,steps,stride,samples,ds,dstatus,df,dp);
        checked(cudaGetLastError(),"launch trace kernel");
    }
    if (!result) checked(cudaEventRecord(event_end),"record end event");
    if (!result) checked(cudaEventSynchronize(event_end),"synchronize owned trace event");
    float milliseconds=0;
    if (!result && checked(cudaEventElapsedTime(&milliseconds,event_start,event_end),"measure trace kernel"))
        timing->kernel_seconds=double(milliseconds)/1000.;
    timing->kernel_wall_seconds=elapsed(stage); stage=Clock::now();
    if (!result) checked(cudaMemcpy(states,ds,state_bytes,cudaMemcpyDeviceToHost),"copy states");
    if (!result) checked(cudaMemcpy(statuses,dstatus,n64*sizeof(int),cudaMemcpyDeviceToHost),"copy statuses");
    if (!result) checked(cudaMemcpy(final,df,n64*3*sizeof(double),cudaMemcpyDeviceToHost),"copy final");
    if (!result) checked(cudaMemcpy(progress,dp,n64*2*sizeof(std::int64_t),cudaMemcpyDeviceToHost),"copy progress");
    timing->device_output_copied=!result;
    timing->device_to_host_seconds=elapsed(stage); stage=Clock::now();
    // Never reset the device or release another caller's allocations/streams.
    for (void* pointer : {static_cast<void*>(da),static_cast<void*>(dg),static_cast<void*>(di),
                         static_cast<void*>(ds),static_cast<void*>(dstatus),static_cast<void*>(df),static_cast<void*>(dp)})
        if (pointer) checked(cudaFree(pointer),"free owned allocation");
    if (event_start) checked(cudaEventDestroy(event_start),"destroy owned start event");
    if (event_end) checked(cudaEventDestroy(event_end),"destroy owned end event");
    timing->free_seconds=elapsed(stage); timing->total_seconds=elapsed(total_start);
    return result;
}
