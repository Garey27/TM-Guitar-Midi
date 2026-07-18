#include "tmgm/cuda_smoke.hpp"

namespace tmgm::native {

bool cuda_backend_compiled() noexcept {
    return false;
}

CudaSmokeResult run_cuda_smoke() {
    CudaSmokeResult result;
    result.detail = "native target was built without CUDA support";
    return result;
}

}  // namespace tmgm::native
