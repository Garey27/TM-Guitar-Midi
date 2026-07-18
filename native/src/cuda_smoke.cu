#include "tmgm/cuda_smoke.hpp"

#include <cuda_runtime.h>

#include <array>
#include <sstream>

namespace tmgm::native {
namespace {

__global__ void increment_kernel(int* values, const int count) {
    const auto index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < count) {
        values[index] += 1;
    }
}

[[nodiscard]] std::string cuda_error(const char* operation, const cudaError_t status) {
    std::ostringstream stream;
    stream << operation << " failed: " << cudaGetErrorString(status);
    return stream.str();
}

}  // namespace

bool cuda_backend_compiled() noexcept {
    return true;
}

CudaSmokeResult run_cuda_smoke() {
    CudaSmokeResult result;
    result.compiled = true;

    auto status = cudaGetDeviceCount(&result.device_count);
    if (status != cudaSuccess) {
        result.detail = cuda_error("cudaGetDeviceCount", status);
        return result;
    }
    if (result.device_count == 0) {
        result.detail = "CUDA runtime found no devices";
        return result;
    }

    cudaDeviceProp properties{};
    status = cudaGetDeviceProperties(&properties, 0);
    if (status != cudaSuccess) {
        result.detail = cuda_error("cudaGetDeviceProperties", status);
        return result;
    }
    result.device_name = properties.name;

    constexpr int kCount = 32;
    std::array<int, kCount> host{};
    for (int index = 0; index < kCount; ++index) {
        host[static_cast<std::size_t>(index)] = index * 3;
    }

    int* device = nullptr;
    status = cudaMalloc(&device, sizeof(int) * kCount);
    if (status != cudaSuccess) {
        result.detail = cuda_error("cudaMalloc", status);
        return result;
    }

    status = cudaMemcpy(device, host.data(), sizeof(int) * kCount, cudaMemcpyHostToDevice);
    if (status == cudaSuccess) {
        increment_kernel<<<1, kCount>>>(device, kCount);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        status = cudaDeviceSynchronize();
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(host.data(), device, sizeof(int) * kCount, cudaMemcpyDeviceToHost);
    }
    const auto free_status = cudaFree(device);

    if (status != cudaSuccess) {
        result.detail = cuda_error("CUDA smoke operation", status);
        return result;
    }
    if (free_status != cudaSuccess) {
        result.detail = cuda_error("cudaFree", free_status);
        return result;
    }
    for (int index = 0; index < kCount; ++index) {
        if (host[static_cast<std::size_t>(index)] != index * 3 + 1) {
            result.detail = "kernel output validation failed";
            return result;
        }
    }

    result.passed = true;
    result.detail = "allocation, host/device copies, kernel launch and synchronization passed";
    return result;
}

}  // namespace tmgm::native
