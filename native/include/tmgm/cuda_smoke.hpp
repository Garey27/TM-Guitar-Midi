#pragma once

#include <string>

namespace tmgm::native {

struct CudaSmokeResult {
    bool compiled = false;
    bool passed = false;
    int device_count = 0;
    std::string device_name;
    std::string detail;
};

[[nodiscard]] bool cuda_backend_compiled() noexcept;
[[nodiscard]] CudaSmokeResult run_cuda_smoke();

}  // namespace tmgm::native
