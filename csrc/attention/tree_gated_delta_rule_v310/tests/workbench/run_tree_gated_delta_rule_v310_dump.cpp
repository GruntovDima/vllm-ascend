// Rendered from run_op_dump.cpp.tmpl: real ACLNN device path, no ICPU substitution.
#include <acl/acl.h>
#include "../../op_host/op_api/aclnn_tree_gated_delta_rule_v310.h"
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static void Check(int code, const char *operation)
{
    if (code != 0) {
        throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(code));
    }
}
static std::vector<uint8_t> Read(const std::string &path, size_t bytes)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || static_cast<size_t>(file.tellg()) != bytes) {
        throw std::runtime_error("Missing or wrong-sized input: " + path);
    }
    file.seekg(0);
    std::vector<uint8_t> data(bytes);
    file.read(reinterpret_cast<char *>(data.data()), bytes);
    if (!file) {
        throw std::runtime_error("Read failed: " + path);
    }
    return data;
}
static void Write(const std::string &path, const std::vector<uint8_t> &data)
{
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char *>(data.data()), data.size());
    if (!file) {
        throw std::runtime_error("Write failed: " + path);
    }
}

struct Tensor {
    void *device = nullptr;
    aclTensor *tensor = nullptr;
    size_t bytes = 0;
    std::vector<uint8_t> original;

    Tensor(const std::vector<int64_t> &shape, aclDataType type)
    {
        std::vector<int64_t> strides(shape.size(), 1);
        size_t elements = 1;
        for (size_t index = shape.size(); index-- > 0;) {
            strides[index] = elements;
            elements *= shape[index];
        }
        bytes = elements * (type == ACL_FLOAT ? 4 : 2);
        Check(aclrtMalloc(&device, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
        tensor = aclCreateTensor(shape.data(), shape.size(), type, strides.data(), 0,
                                 ACL_FORMAT_ND, shape.data(), shape.size(), device);
        if (tensor == nullptr) {
            aclrtFree(device);
            device = nullptr;
            throw std::runtime_error("aclCreateTensor failed");
        }
    }
    ~Tensor()
    {
        if (tensor) {
            aclDestroyTensor(tensor);
        }
        if (device) {
            aclrtFree(device);
        }
    }
    Tensor(const Tensor &) = delete;
    void Load(const std::string &path)
    {
        original = Read(path, bytes);
        Check(aclrtMemcpy(device, bytes, original.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "H2D");
    }
    std::vector<uint8_t> Host() const
    {
        std::vector<uint8_t> data(bytes);
        Check(aclrtMemcpy(data.data(), bytes, device, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "D2H");
        return data;
    }
};

int main(int argc, char **argv)
{
    if (argc != 4) {
        std::cerr << "Usage: runner CASE_DIR LOGICAL_DEVICE REPEATS\n";
        return 2;
    }
    const std::string directory = argv[1];
    const int deviceId = std::stoi(argv[2]), repeats = std::stoi(argv[3]);
    // A one-launch probe isolates sanitizer/runtime issues. The correctness
    // suite explicitly requests and verifies three repeats on every shape.
    if (repeats < 1 || deviceId < 0) {
        return 2;
    }
    aclrtStream stream = nullptr;
    bool initialized = false, deviceSet = false;
    try {
        int64_t n, hk, hv, tile;
        float scale;
        std::ifstream params(directory + "/params.txt");
        params >> n >> hk >> hv >> tile >> scale;
        if (!params || n < 1 || n > 65 || hk < 1 || hv < hk || hv > 32) {
            throw std::runtime_error("Invalid params.txt");
        }
        std::vector<int64_t> parents(n);
        for (auto &parent : parents) {
            params >> parent;
        }
        if (!params) {
            throw std::runtime_error("Incomplete parents");
        }
        // ACLNN sanitizer runs require lite_exception metadata to associate
        // tensor allocations with the launched kernel. Ordinary runs omit it.
        Check(aclInit(std::getenv("TREE_GDN_ACL_CONFIG")), "aclInit");
        initialized = true;
        Check(aclrtSetDevice(deviceId), "aclrtSetDevice");
        deviceSet = true;
        Check(aclrtCreateStream(&stream), "aclrtCreateStream");
        {
            const std::vector<std::vector<int64_t>> shapes = {
                {n, hk, 128}, {n, hk, 128}, {n, hv, 128}, {n, hv},
                {hv, 128, 128}, {n, hv}, {n, hv, 128}, {n, hv, 128, 128}};
            const std::vector<std::string> names = {
                "query", "key", "value", "beta", "initial_state", "g", "out", "snapshots"};
            std::vector<std::unique_ptr<Tensor>> tensors;
            for (size_t index = 0; index < shapes.size(); ++index) {
                tensors.emplace_back(new Tensor(shapes[index], index == 5 ? ACL_FLOAT : ACL_FLOAT16));
                if (index < 6) {
                    tensors.back()->Load(directory + "/input_" + names[index] + ".bin");
                }
            }
            auto *parentArray = aclCreateIntArray(parents.data(), parents.size());
            if (!parentArray) {
                throw std::runtime_error("aclCreateIntArray failed");
            }
            // Distinct base addresses, but out overlaps the snapshots allocation.
            const std::vector<int64_t> outStrides = {hv * 128, 128, 1};
            auto *overlap = aclCreateTensor(shapes[6].data(), 3, ACL_FLOAT16,
                outStrides.data(), 0, ACL_FORMAT_ND, shapes[6].data(), 3,
                static_cast<uint8_t *>(tensors[7]->device) + 32);
            if (!overlap) throw std::runtime_error("overlap tensor");
            uint64_t badWorkspace = 0;
            aclOpExecutor *badExecutor = nullptr;
            const auto badStatus = aclnnTreeGatedDeltaRuleV310GetWorkspaceSize(
                tensors[0]->tensor, tensors[1]->tensor, tensors[2]->tensor,
                tensors[3]->tensor, tensors[4]->tensor, tensors[5]->tensor,
                parentArray, scale, tile, overlap, tensors[7]->tensor,
                &badWorkspace, &badExecutor);
            aclDestroyTensor(overlap);
            if (badStatus == 0 || badExecutor != nullptr) {
                throw std::runtime_error("Partial-overlap output was not rejected");
            }
            std::cout << "OK: partial-overlap output rejected" << std::endl;
            for (int repeat = 0; repeat < repeats; ++repeat) {
                for (size_t index : {6U, 7U}) {
                    Check(aclrtMemset(tensors[index]->device, tensors[index]->bytes,
                                      0x31 + repeat * 37, tensors[index]->bytes), "poison outputs");
                }
                uint64_t workspaceSize = 0;
                aclOpExecutor *executor = nullptr;
                Check(aclnnTreeGatedDeltaRuleV310GetWorkspaceSize(
                    tensors[0]->tensor, tensors[1]->tensor, tensors[2]->tensor,
                    tensors[3]->tensor, tensors[4]->tensor, tensors[5]->tensor,
                    parentArray, scale, tile, tensors[6]->tensor, tensors[7]->tensor,
                    &workspaceSize, &executor), "tree GetWorkspaceSize");
                void *workspace = nullptr;
                if (workspaceSize) {
                    Check(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST), "workspace malloc");
                }
                const auto launchStatus = aclnnTreeGatedDeltaRuleV310(workspace, workspaceSize, executor, stream);
                const auto syncStatus = aclrtSynchronizeStream(stream);
                if (workspace) {
                    aclrtFree(workspace);
                }
                Check(launchStatus, "tree launch");
                Check(syncStatus, "tree synchronize");
                for (size_t index : {6U, 7U}) {
                    Write(directory + "/output_" + names[index] + "_r" + std::to_string(repeat) + ".bin",
                          tensors[index]->Host());
                }
                std::cout << "DUMP repeat=" << repeat << " workspace=" << workspaceSize << std::endl;
            }
            aclDestroyIntArray(parentArray);
            for (size_t index = 0; index < 6; ++index) {
                if (tensors[index]->Host() != tensors[index]->original) {
                    throw std::runtime_error("Input mutated: " + names[index]);
                }
            }
        }
        Check(aclrtDestroyStream(stream), "aclrtDestroyStream");
        stream = nullptr;
        Check(aclrtResetDevice(deviceId), "aclrtResetDevice");
        deviceSet = false;
        Check(aclFinalize(), "aclFinalize");
        std::cout << "OK: device outputs dumped; inputs unchanged\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        if (stream) {
            aclrtDestroyStream(stream);
        }
        if (deviceSet) {
            aclrtResetDevice(deviceId);
        }
        if (initialized) {
            aclFinalize();
        }
        return 1;
    }
}
