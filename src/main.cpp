#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../include/backend.h"
#include "../include/chat.h"
#include "../include/context.h"
#include "../include/cuda_runtime.h"
#include "../include/debug.h"
#include "../include/forward.h"
#include "../include/gguf.h"
#include "../include/gguf_loader.h"
#include "../include/gguf_tokenizer.h"
#include "../include/loader.h"
#include "../include/model.h"
#include "../include/ops.h"
#include "../include/promptbuilder.h"
#include "../include/quant.h"
#include "../include/request_context.h"
#include "../include/sampler.h"
#include "../include/terminal.h"
#include "../include/thread_pool.h"
#include "../include/tokenizer.h"

namespace LLaMA_Tensor {

// ---------------------------------------------------------------------------
// Argument parsing helpers
// ---------------------------------------------------------------------------
static bool ParseIntArg(const char* text, int& value) {
  char* end = nullptr;
  int64_t parsed = std::strtoll(text, &end, 10);
  if (end == text || *end != '\0') {
    return false;
  }
  if (parsed < 0 || parsed > 1000000) {
    return false;
  }
  value = static_cast<int>(parsed);
  return true;
}

static bool ParseFloatArg(const char* text, float& value) {
  char* end = nullptr;
  errno = 0;
  double parsed = std::strtod(text, &end);
  if (end == text || *end != '\0') {
    return false;
  }
  if (errno == ERANGE || !std::isfinite(parsed) || parsed < 0.0 ||
      parsed > 10000.0) {
    return false;
  }
  value = static_cast<float>(parsed);
  return true;
}

static bool ParseUintArg(const char* text, unsigned int& value) {
  if (text[0] == '-' || text[0] == '\0') {
    return false;
  }
  char* end = nullptr;
  errno = 0;
  uint64_t parsed = std::strtoull(text, &end, 10);
  if (end == text || *end != '\0') {
    return false;
  }
  if (errno == ERANGE || parsed > std::numeric_limits<unsigned int>::max()) {
    return false;
  }
  value = static_cast<unsigned int>(parsed);
  return true;
}

static bool ParseBackendArg(const char* text, BackendConfig& config) {
  BackendKind kind;
  if (!ParseBackendKind(text, kind)) {
    return false;
  }
  config.kind = kind;
  return true;
}

static bool ParseDeviceArg(const char* text, BackendConfig& config) {
  int device_id = 0;
  if (!ParseIntArg(text, device_id) || device_id < 0) {
    return false;
  }
  config.device_id = device_id;
  config.device_id_set = true;
  return true;
}

// Human-readable note describing where the selected backend executes.
static std::string BackendNoteText(const BackendConfig& config) {
  return BackendExecutionNote(config);
}

static bool ValidateBackendOrPrint(const BackendConfig& config) {
  try {
    ValidateBackend(config);
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Backend setup failed: " << e.what() << "\n";
    return false;
  }
}

static void PrintBackendInfo(const BackendConfig& config) {
  std::cout << "backend: " << BackendKindName(config.kind) << "\n";
  if (config.kind == BackendKind::kCuda) {
    std::cout << "cuda: " << CudaDeviceSummary(config.device_id) << "\n";
  }
  std::cout << BackendNoteText(config) << "\n";
}

// ---------------------------------------------------------------------------
// CUDA backend helpers
// ---------------------------------------------------------------------------
static std::string FormatMb(size_t bytes) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(2)
      << (static_cast<double>(bytes) / (1024.0 * 1024.0)) << " MB";
  return out.str();
}

static bool PrepareCudaWeightsOrPrint(MiniLlamaModel& model,
                                      const BackendConfig& config) {
  if (config.kind != BackendKind::kCuda) {
    return true;
  }
  try {
    UploadModelWeightsToCuda(model, config.device_id);
    return true;
  } catch (const std::exception& e) {
    std::cerr << "CUDA weight upload failed: " << e.what() << "\n";
    return false;
  }
}

static void PrintCudaWeightSummary(const MiniLlamaModel& model,
                                   const std::string& indent = "") {
  std::cout << indent
            << "uploaded weights: " << ModelCudaUploadedWeightCount(model)
            << "\n";
  std::cout << indent << "gpu memory used: " << ModelCudaMemoryBytes(model)
            << " bytes (" << FormatMb(ModelCudaMemoryBytes(model)) << ")\n";
}

struct LinearWeightTypeCounts {
  size_t f32 = 0;
  size_t q8_0 = 0;
  size_t q4_0 = 0;
  size_t q4_1 = 0;

  size_t total() const { return f32 + q8_0 + q4_0 + q4_1; }

  size_t quantized() const { return q8_0 + q4_0 + q4_1; }
};

static void CountLinearWeight(const QuantizedTensor& weight,
                              LinearWeightTypeCounts& counts) {
  switch (weight.type) {
    case QuantType::kF32:
      counts.f32 += 1;
      return;
    case QuantType::kQ80:
      counts.q8_0 += 1;
      return;
    case QuantType::kQ40:
      counts.q4_0 += 1;
      return;
    case QuantType::kQ41:
      counts.q4_1 += 1;
      return;
  }
}

static LinearWeightTypeCounts ModelLinearWeightTypeCounts(
    const MiniLlamaModel& model) {
  LinearWeightTypeCounts counts;
  CountLinearWeight(model.lm_head, counts);
  for (const auto& lw : model.layers) {
    CountLinearWeight(lw.wq, counts);
    CountLinearWeight(lw.wk, counts);
    CountLinearWeight(lw.wv, counts);
    CountLinearWeight(lw.wo, counts);
    CountLinearWeight(lw.w_gate, counts);
    CountLinearWeight(lw.w_up, counts);
    CountLinearWeight(lw.w_down, counts);
  }
  return counts;
}

static void PrintCudaExecutionSummary(const MiniLlamaModel& model,
                                      const std::string& indent = "") {
  PrintCudaWeightSummary(model, indent);
  LinearWeightTypeCounts counts = ModelLinearWeightTypeCounts(model);
  size_t uploaded_f32 = ModelCudaUploadedF32WeightCount(model);
  size_t uploaded_q8_0 = ModelCudaUploadedQ80WeightCount(model);
  size_t uploaded_q4_0 = ModelCudaUploadedQ40WeightCount(model);
  size_t uploaded_q4_1 = ModelCudaUploadedQ41WeightCount(model);
  std::cout << indent << "cuda f32 Linear weights: " << uploaded_f32 << "/"
            << counts.total() << "\n";
  if (counts.q8_0 > 0) {
    std::cout << indent << "cuda q8_0 Linear weights: " << uploaded_q8_0 << "/"
              << counts.q8_0 << "\n";
  }
  if (counts.q4_0 > 0) {
    std::cout << indent << "cuda q4_0 Linear weights: " << uploaded_q4_0 << "/"
              << counts.q4_0 << "\n";
  }
  if (counts.q4_1 > 0) {
    std::cout << indent << "cuda q4_1 Linear weights: " << uploaded_q4_1 << "/"
              << counts.q4_1 << "\n";
  }
  size_t q8_0_fallback =
      counts.q8_0 >= uploaded_q8_0 ? counts.q8_0 - uploaded_q8_0 : 0;
  size_t q4_0_fallback =
      counts.q4_0 >= uploaded_q4_0 ? counts.q4_0 - uploaded_q4_0 : 0;
  size_t q4_1_fallback =
      counts.q4_1 >= uploaded_q4_1 ? counts.q4_1 - uploaded_q4_1 : 0;
  if (q8_0_fallback + q4_0_fallback + q4_1_fallback > 0) {
    std::cout << indent
              << "cuda fallback: unsupported or missing quantized Linear "
                 "weights use CPU path"
              << " (Q8_0=" << q8_0_fallback << ", Q4_0=" << q4_0_fallback
              << ", Q4_1=" << q4_1_fallback << ")\n";
  }
}

static void PrintCudaRuntimeSummary(const MiniLlamaModel& model,
                                    const std::string& indent = "") {
  std::cout << indent << "cuda Linear calls: " << ModelCudaLinearCalls(model)
            << "\n";
  std::cout << indent
            << "cuda activation calls: " << ModelCudaActivationCalls(model)
            << "\n";
  std::cout << indent
            << "cuda attention calls: " << ModelCudaAttentionCalls(model)
            << "\n";
  std::cout << indent << "cpu attention fallback calls: "
            << ModelCudaAttentionCpuFallbacks(model) << "\n";
  std::cout << indent
            << "cuda kv write bytes: " << ModelCudaKvCacheWriteBytes(model)
            << "\n";
  std::cout << indent
            << "cuda kv read bytes: " << ModelCudaKvCacheReadBytes(model)
            << "\n";
  std::cout << indent
            << "host->device copies: " << ModelCudaHostToDeviceCopies(model)
            << " (" << ModelCudaHostToDeviceBytes(model) << " bytes)\n";
  std::cout << indent
            << "device->host copies: " << ModelCudaDeviceToHostCopies(model)
            << " (" << ModelCudaDeviceToHostBytes(model) << " bytes)\n";
}

namespace {

Tensor DequantizeLinearWeight(const QuantizedTensor& weight) {
  switch (weight.type) {
    case QuantType::kF32:
      return QTensor2Tensor(weight);
    case QuantType::kQ80:
      return DequantizedFromQ80(weight.q80_data, weight.shape);
    case QuantType::kQ40:
      return DequantizedFromQ40(weight.q40_data, weight.shape);
    case QuantType::kQ41:
      return DequantizedFromQ41(weight.q41_data, weight.shape);
  }
  throw std::runtime_error("unknown quantized weight type");
}

void QuantizeLinearWeight(QuantizedTensor& weight, QuantType target) {
  Tensor f32 = DequantizeLinearWeight(weight);
  QuantizedTensor quantized;
  quantized.shape = f32.shape;
  switch (target) {
    case QuantType::kQ80:
      quantized.type = QuantType::kQ80;
      quantized.q80_data = QuantizedToQ80(f32);
      break;
    case QuantType::kQ40:
      quantized.type = QuantType::kQ40;
      quantized.q40_data = QuantizedToQ40(f32);
      break;
    case QuantType::kQ41:
      quantized.type = QuantType::kQ41;
      quantized.q41_data = QuantizedToQ41(f32);
      break;
    case QuantType::kF32:
      quantized = Tensor2QTensor(f32);
      break;
  }
  weight = std::move(quantized);
}

// Model-level quantization helper. The corresponding QuantizeModelToQ80/Q40
// helpers do not exist in the project; only the per-tensor functions declared
// in include/quant.h are available, so all Linear weights are visited here.
void QuantizeModelWeights(MiniLlamaModel& model, QuantType target) {
  QuantizeLinearWeight(model.lm_head, target);
  for (LayerWeights& layer : model.layers) {
    QuantizeLinearWeight(layer.wq, target);
    QuantizeLinearWeight(layer.wk, target);
    QuantizeLinearWeight(layer.wv, target);
    QuantizeLinearWeight(layer.wo, target);
    QuantizeLinearWeight(layer.w_gate, target);
    QuantizeLinearWeight(layer.w_up, target);
    QuantizeLinearWeight(layer.w_down, target);
  }
}

std::string TensorShapeString(const Tensor& tensor) {
  std::string shape = "[";
  for (size_t i = 0; i < tensor.shape.size(); ++i) {
    if (i > 0) {
      shape += ", ";
    }
    shape += std::to_string(tensor.shape[i]);
  }
  shape += "]";
  return shape;
}

size_t F32Bytes(const Tensor& tensor) {
  return tensor.TensorNum() * sizeof(float);
}

size_t F32Bytes(const QuantizedTensor& tensor) {
  return tensor.ele_num() * sizeof(float);
}

// F32-equivalent weight footprint, used by the benchmark memory report.
// ModelWeightBytesF32() itself is not declared by the project headers.
size_t ModelWeightBytesF32(const MiniLlamaModel& model) {
  size_t bytes = 0;
  bytes += F32Bytes(model.token_embedding);
  bytes += F32Bytes(model.lm_head);
  bytes += F32Bytes(model.final_norm);
  for (const LayerWeights& lw : model.layers) {
    bytes += F32Bytes(lw.attention_norm);
    bytes += F32Bytes(lw.bk);
    bytes += F32Bytes(lw.bq);
    bytes += F32Bytes(lw.bv);
    bytes += F32Bytes(lw.ffn_norm);
    bytes += F32Bytes(lw.w_down);
    bytes += F32Bytes(lw.w_gate);
    bytes += F32Bytes(lw.w_up);
    bytes += F32Bytes(lw.wk);
    bytes += F32Bytes(lw.wo);
    bytes += F32Bytes(lw.wq);
    bytes += F32Bytes(lw.wv);
  }
  return bytes;
}

}  // namespace

static void ApplyQuantOverride(MiniLlamaModel& model,
                               const std::string& quant_type) {
  if (quant_type.empty()) {
    return;
  }
  if (quant_type == "q8_0") {
    QuantizeModelWeights(model, QuantType::kQ80);
    return;
  }
  if (quant_type == "q4_0") {
    QuantizeModelWeights(model, QuantType::kQ40);
    return;
  }
  throw std::runtime_error("unsupported quant type: " + quant_type);
}

static Tensor RunLogitsForTokens(const MiniLlamaModel& model,
                                 const std::vector<int>& tokens) {
  Context ctx(&model);
  Batch batch = Batch::Prefill(tokens, 0);
  return ForwardBatch(ctx, model, batch);
}

static std::pair<float, float> LogitsError(const Tensor& baseline,
                                           const Tensor& candidate) {
  if (baseline.shape != candidate.shape) {
    throw std::runtime_error(
        "LogitsError: shape mismatch, baseline=" + TensorShapeString(baseline) +
        ", candidate=" + TensorShapeString(candidate));
  }
  float max_err = 0.0f;
  float sum_err = 0.0f;
  for (size_t i = 0; i < baseline.data.size(); ++i) {
    float err = std::abs(baseline.data[i] - candidate.data[i]);
    max_err = std::max(max_err, err);
    sum_err += err;
  }
  return {max_err,
          sum_err / static_cast<float>(baseline.data.size())};
}

// ---------------------------------------------------------------------------
// Generate mode
// ---------------------------------------------------------------------------
static void PrintGenerateUsage(const char* prog) {
  std::cout
      << "Usage: " << prog << " generate [options]\n"
      << "Options:\n"
      << "  --model <path|dir>   Path to model weights binary or model "
         "directory (default: models/tiny/model.bin)\n"
      << "  --config <path>      Path to model config JSON (default: "
         "models/tiny/model.json)\n"
      << "  -p, --prompt <str>   Input prompt text (default: \"hello\")\n"
      << "  -n, --n-predict <n>  Number of tokens to generate (default: 16)\n"
      << "  --temperature <T>    Sampling temperature (default: 0.0 = greedy)\n"
      << "  --top-k <k>          Top-k sampling (default: 0 = disabled)\n"
      << "  --seed <S>           Random seed for reproducible sampling "
         "(default: 0 = random)\n"
      << "  --tokenizer <path>   Path to vocab.json tokenizer file\n"
      << "  --quant q8_0|q4_0    Quantize loaded Linear weights before "
         "generation\n"
      << "  --threads <n>        Number of threads for parallel ops (0 = "
         "auto)\n"
      << "  --backend cpu|cuda  Execution backend (default: cpu; cuda requires "
         "-DMINI_LLAMA_CUDA=ON)\n"
      << "  --device <n>        CUDA device id for --backend cuda (default: 0)\n"
      << "  --dump-logits <dir>  Dump logits for each step to directory\n"
      << "  -h, --help           Show this help\n";
}

static void DumpLogits(const Tensor& logits, const std::string& path) {
  std::ofstream out(path, std::ios::binary);
  if (!out.is_open()) {
    throw std::runtime_error("failed to open logits dump file: " + path);
  }
  out.write(reinterpret_cast<const char*>(logits.data.data()),
            static_cast<std::streamsize>(logits.data.size() * sizeof(float)));
  if (!out.good()) {
    throw std::runtime_error("failed to write logits to: " + path);
  }
}

static void EnsureDumpDirectory(const std::string& path) {
  std::error_code error;
  std::filesystem::create_directories(path, error);
  if (error) {
    throw std::runtime_error("failed to create logits dump directory: " + path +
                             ": " + error.message());
  }
  if (!std::filesystem::is_directory(path)) {
    throw std::runtime_error("logits dump path is not a directory: " + path);
  }
}

static void DumpGeneratedTokens(const std::vector<int>& generated,
                                const std::string& path) {
  std::ofstream out(path);
  if (!out.is_open()) {
    throw std::runtime_error("failed to open generation token dump file: " +
                             path);
  }
  for (size_t i = 0; i < generated.size(); ++i) {
    if (i > 0) {
      out << " ";
    }
    out << generated[i];
  }
  out << "\n";
  if (!out.good()) {
    throw std::runtime_error("failed to write generation tokens to: " + path);
  }
}

// ---------------------------------------------------------------------------
// Tokenizer path resolution
// ---------------------------------------------------------------------------

static std::string ResolveManifestTokenizerPath(
    const std::string& config_path) {
  try {
    ModelManifest manifest = ParseManifest(config_path);
    if (manifest.tokenizer.type != "json_vocab" ||
        manifest.tokenizer.path.empty()) {
      return "";
    }
    std::filesystem::path tokenizer_path(manifest.tokenizer.path);
    if (tokenizer_path.is_relative()) {
      std::filesystem::path config_file(config_path);
      tokenizer_path = config_file.parent_path() / tokenizer_path;
    }
    return tokenizer_path.string();
  } catch (const std::exception&) {
    return "";
  }
}

// ---------------------------------------------------------------------------
// Model loading helper: supports both JSON+BIN and GGUF
// ---------------------------------------------------------------------------

struct LoadedModel {
  MiniLlamaModel model;
  std::unique_ptr<Tokenizer> tokenizer;
  std::string chat_template;
};

static bool IsGgufFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) {
    return false;
  }
  char magic[4];
  return f.read(magic, 4) && std::memcmp(magic, "GGUF", 4) == 0;
}

static std::string FindGgufInDirectory(const std::string& path) {
  std::vector<std::filesystem::path> candidates;
  std::error_code ec;
  for (const auto& entry : std::filesystem::directory_iterator(path, ec)) {
    if (entry.is_regular_file() && entry.path().extension() == ".gguf") {
      candidates.push_back(entry.path());
    }
  }
  if (candidates.empty()) {
    return "";
  }
  std::sort(candidates.begin(), candidates.end());
  return candidates.front().string();
}

static std::unique_ptr<Tokenizer> CreateTokenizerFromVocabHint(
    const std::string& vocab_path) {
  std::filesystem::path vocab(vocab_path);
  std::filesystem::path dir = vocab.parent_path();
  std::filesystem::path merges = dir / "merges.txt";
  std::filesystem::path special = dir / "special_tokens.json";
  if (std::filesystem::exists(merges)) {
    return CreateBpeTokenizer(vocab.string(), merges.string(),
                              special.string());
  }
  return CreateJsonTokenizer(vocab.string());
}

// Load the chat template stored in GGUF metadata. If the key is missing, fall
// back to a chat_template.txt placed next to the model file.
static std::string LoadChatTemplateFromGguf(const std::string& gguf_path) {
  std::string chat_template;
  if (GgufGetMetadataString(gguf_path, "tokenizer.chat_template",
                            chat_template)) {
    return chat_template;
  }
  std::filesystem::path sidecar =
      std::filesystem::path(gguf_path).parent_path() / "chat_template.txt";
  if (!std::filesystem::exists(sidecar)) {
    return "";
  }
  std::ifstream in(sidecar);
  if (!in) {
    return "";
  }
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

static LoadedModel LoadModelAndTokenizer(
    const std::string& path, const std::string& explicit_config_path = "",
    const std::string& explicit_tokenizer_path = "") {
  LoadedModel result;

  std::string model_path = path;
  std::string config_path;
  std::string tokenizer_path;
  if (std::filesystem::is_directory(path)) {
    std::filesystem::path bin_path = std::filesystem::path(path) / "model.bin";
    std::filesystem::path json_path =
        std::filesystem::path(path) / "model.json";
    std::string gguf_path = FindGgufInDirectory(path);
    if ((!std::filesystem::exists(bin_path) ||
         !std::filesystem::exists(json_path)) &&
        !gguf_path.empty()) {
      model_path = gguf_path;
    }
  }

  bool is_gguf = IsGgufFile(model_path);

  if (is_gguf) {
    result.model = LoadGgufModel(model_path);
    if (!result.model.loaded) {
      return result;
    }

    // 1. Try to Load tokenizer from GGUF metadata (M14)
    if (!explicit_tokenizer_path.empty()) {
      result.tokenizer = CreateTokenizerFromVocabHint(explicit_tokenizer_path);
    } else {
      result.tokenizer = CreateGgufTokenizer(model_path);
    }

    // 2. Fallback to external vocab.json + merges.txt if GGUF has no tokenizer
    // metadata
    if (!result.tokenizer) {
      std::filesystem::path gguf_dir =
          std::filesystem::path(model_path).parent_path();
      std::string vocab_path = (gguf_dir / "vocab.json").string();
      std::string merges_path = (gguf_dir / "merges.txt").string();
      std::string special_path = (gguf_dir / "special_tokens.json").string();
      if (std::filesystem::exists(vocab_path) &&
          std::filesystem::exists(merges_path)) {
        result.tokenizer =
            CreateBpeTokenizer(vocab_path, merges_path, special_path);
      }
    }

    // 3. Load chat template from GGUF metadata (M14)
    result.chat_template = LoadChatTemplateFromGguf(model_path);
  } else {
    // Directory-based JSON+BIN format
    if (std::filesystem::is_directory(path)) {
      model_path = path + "/model.bin";
      config_path = path + "/model.json";
    } else {
      model_path = path;
      if (!explicit_config_path.empty()) {
        config_path = explicit_config_path;
      } else {
        config_path =
            std::filesystem::path(path).parent_path().string() + "/model.json";
      }
    }
    result.model = LoadModel(config_path, model_path);
    if (!result.model.loaded) {
      return result;
    }
    if (!explicit_tokenizer_path.empty()) {
      tokenizer_path = explicit_tokenizer_path;
    }
    if (tokenizer_path.empty()) {
      std::filesystem::path dir =
          std::filesystem::is_directory(path)
              ? std::filesystem::path(path)
              : std::filesystem::path(path).parent_path();
      std::filesystem::path auto_vocab = dir / "vocab.json";
      if (std::filesystem::exists(auto_vocab)) {
        tokenizer_path = auto_vocab.string();
      }
    }
    if (tokenizer_path.empty()) {
      tokenizer_path = ResolveManifestTokenizerPath(config_path);
    }
    result.tokenizer = CreateJsonTokenizer(tokenizer_path);
  }

  if (!result.tokenizer) {
    result.model.load_error = "Failed to load tokenizer";
    result.model.loaded = false;
    return result;
  }

  return result;
}

static size_t CommonPrefixLength(const std::vector<int>& a,
                                 const std::vector<int>& b) {
  size_t n = std::min(a.size(), b.size());
  size_t i = 0;
  while (i < n && a[i] == b[i]) {
    ++i;
  }
  return i;
}

static void PrintRequestTrace(const RequestContext& request,
                              std::ostream& out = std::cout) {
  for (const std::string& line : FormatRequestTraceEvents(request)) {
    out << line << "\n";
  }
  out << FormatRequestTraceSummary(request) << "\n";
}

static int RunGenerate(int argc, char** argv) {
  std::string model_path = "models/tiny/model.bin";
  std::string config_path = "models/tiny/model.json";
  bool config_path_set = false;
  std::string prompt = "hello";
  int n_predict = 16;
  float temperature = 0.0f;
  int top_k = 0;
  unsigned int seed = 0;
  std::string dump_logits_dir;
  std::string tokenizer_path;
  std::string quant_type;
  int n_threads = 0;
  BackendConfig backend_config;

  for (int i = 2; i < argc; ++i) {
    if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
      model_path = argv[++i];
    } else if (std::strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
      config_path = argv[++i];
      config_path_set = true;
    } else if ((std::strcmp(argv[i], "-p") == 0 ||
                std::strcmp(argv[i], "--prompt") == 0) &&
               i + 1 < argc) {
      prompt = argv[++i];
    } else if ((std::strcmp(argv[i], "-n") == 0 ||
                std::strcmp(argv[i], "--n-predict") == 0) &&
               i + 1 < argc) {
      if (!ParseIntArg(argv[++i], n_predict)) {
        std::cerr
            << "Invalid --n-predict value. Expected a non-negative integer.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
      if (!ParseFloatArg(argv[++i], temperature)) {
        std::cerr
            << "Invalid --temperature value. Expected a non-negative float.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
      if (!ParseIntArg(argv[++i], top_k)) {
        std::cerr
            << "Invalid --top-k value. Expected a non-negative integer.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
      if (!ParseUintArg(argv[++i], seed)) {
        std::cerr << "Invalid --seed value. Expected a non-negative integer.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--dump-logits") == 0 && i + 1 < argc) {
      dump_logits_dir = argv[++i];
    } else if (std::strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
      tokenizer_path = argv[++i];
    } else if (std::strcmp(argv[i], "--quant") == 0 && i + 1 < argc) {
      quant_type = argv[++i];
    } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      if (!ParseIntArg(argv[++i], n_threads) || n_threads < 0) {
        std::cerr << "Invalid --threads value.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
      if (!ParseBackendArg(argv[++i], backend_config)) {
        std::cerr << "Invalid --backend value. Supported values: cpu, cuda.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
      if (!ParseDeviceArg(argv[++i], backend_config)) {
        std::cerr
            << "Invalid --device value. Expected a non-negative integer.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "-h") == 0 ||
               std::strcmp(argv[i], "--help") == 0) {
      PrintGenerateUsage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown or incomplete argument: " << argv[i] << "\n";
      PrintGenerateUsage(argv[0]);
      return 1;
    }
  }

  if (!quant_type.empty() && quant_type != "q8_0" && quant_type != "q4_0") {
    std::cerr << "Invalid --quant value: " << quant_type
              << ". Supported values: q8_0, q4_0.\n";
    return 1;
  }

  if (!ValidateBackendOrPrint(backend_config)) {
    return 1;
  }

  std::cout << "mini-llama.cpp\n";
  std::cout << "==============\n\n";
  PrintBackendInfo(backend_config);

  if (!dump_logits_dir.empty()) {
    try {
      EnsureDumpDirectory(dump_logits_dir);
    } catch (const std::exception& e) {
      std::cerr << "Logits dump setup failed: " << e.what() << "\n";
      return 1;
    }
  }

  // For backward compat, if --config was explicitly set but --model is the
  // default, override model_path to be the config's directory.
  if (config_path_set && model_path == "models/tiny/model.bin") {
    model_path = std::filesystem::path(config_path).parent_path().string() +
                 "/model.bin";
  }

  RequestContext request =
      StartRequest("generate", BackendKindName(backend_config.kind), model_path);

  auto stage_start = RequestClock::now();
  LoadedModel lm = LoadModelAndTokenizer(
      model_path, config_path_set ? config_path : "", tokenizer_path);
  request.model_load_ms = ElapsedMs(stage_start);
  request.RecordEvent("model_load", request.model_load_ms, 0, model_path);
  if (!lm.model.loaded) {
    request.SetError("Failed to load model: " + lm.model.load_error);
    request.Finish();
    PrintRequestTrace(request, std::cerr);
    std::cerr << request.error << "\n";
    return 1;
  }
  if (!lm.tokenizer) {
    request.SetError("Failed to load tokenizer.");
    request.Finish();
    PrintRequestTrace(request, std::cerr);
    std::cerr << request.error << "\n";
    return 1;
  }
  if (lm.model.config.vocab_size < lm.tokenizer->vocab_size()) {
    request.SetError("Model vocab_size must be at least " +
                     std::to_string(lm.tokenizer->vocab_size()) +
                     " for the tokenizer.");
    request.Finish();
    PrintRequestTrace(request, std::cerr);
    std::cerr << request.error << "\n";
    return 1;
  }
  MiniLlamaModel& model = lm.model;
  try {
    stage_start = RequestClock::now();
    ApplyQuantOverride(model, quant_type);
    request.RecordEvent("quantize", ElapsedMs(stage_start), 0,
                        quant_type.empty() ? "model-native" : quant_type);
  } catch (const std::exception& e) {
    request.SetError("Quantization failed: " + std::string(e.what()));
    request.Finish();
    PrintRequestTrace(request, std::cerr);
    std::cerr << request.error << "\n";
    return 1;
  }
  if (!PrepareCudaWeightsOrPrint(model, backend_config)) {
    request.SetError("CUDA weight preparation failed");
    request.Finish();
    PrintRequestTrace(request, std::cerr);
    return 1;
  }
  std::unique_ptr<Tokenizer>& tokenizer = lm.tokenizer;
  stage_start = RequestClock::now();
  std::vector<int> tokens = tokenizer->Encode(prompt);
  request.tokenize_ms = ElapsedMs(stage_start);
  request.prompt_tokens = static_cast<int>(tokens.size());
  request.RecordEvent("tokenize", request.tokenize_ms, request.prompt_tokens,
                      "prompt");
  std::cout << "prompt: " << prompt << "\n";
  std::cout << "tokens: [";
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << tokens[i];
  }
  std::cout << "]\n";
  SetThreadCount(n_threads);
  std::cout << "sampling: temperature=" << temperature << ", top_k=" << top_k
            << ", seed=" << seed << "\n";
  std::cout << "quant: " << (quant_type.empty() ? "model-native" : quant_type)
            << "\n";
  std::cout << "threads: " << GetThreadCount() << "\n\n";
  if (backend_config.kind == BackendKind::kCuda) {
    PrintCudaExecutionSummary(model);
    std::cout << "\n";
  }

  if (tokens.size() > static_cast<size_t>(model.config.max_seq_len)) {
    request.SetError("Prompt is too long for max_seq_len=" +
                     std::to_string(model.config.max_seq_len) + ".");
    request.Finish();
    PrintRequestTrace(request, std::cerr);
    std::cerr << request.error << "\n";
    return 1;
  }
  if (tokens.size() + static_cast<size_t>(n_predict) >
      static_cast<size_t>(model.config.max_seq_len)) {
    request.SetError("Requested tokens exceed context window.");
    request.Finish();
    PrintRequestTrace(request, std::cerr);
    std::cerr << request.error << "\n";
    return 1;
  }

  Context ctx(&model);
  SamplerParams sampling_params;
  sampling_params.temperature = temperature;
  sampling_params.top_k = top_k;
  sampling_params.seed = seed;
  Sampler sampler(sampling_params);

  size_t prompt_len = tokens.size();
  int step = 0;
  try {
    Tensor logits;
    std::cout << "prefill...\n";
    stage_start = RequestClock::now();
    if (!dump_logits_dir.empty()) {
      for (size_t i = 0; i < prompt_len; ++i) {
        Batch prefill_step =
            Batch::Prefill({tokens[i]}, static_cast<int>(i));
        logits = ForwardBatch(ctx, model, prefill_step);
        ++ctx.n_prefill_tokens;
        DumpLogits(logits, dump_logits_dir + "/logits_step" +
                               std::to_string(step) + ".bin");
        ++step;
      }
    } else {
      Batch prefill = Batch::Prefill(tokens, 0);
      logits = ForwardBatch(ctx, model, prefill);
      ctx.n_prefill_tokens += static_cast<int>(tokens.size());
    }
    request.prefill_ms = ElapsedMs(stage_start);
    request.prefill_tokens = static_cast<int>(prompt_len);
    request.RecordEvent("prefill", request.prefill_ms, request.prefill_tokens,
                        dump_logits_dir.empty() ? "batch" : "step_dump");

    if (n_predict > 0) {
      stage_start = RequestClock::now();
      int next_token = sampler.Sample(logits);
      request.sample_ms += ElapsedMs(stage_start);
      tokens.push_back(next_token);

      std::cout << "decode loop...\n";
      for (int i = 1; i < n_predict; ++i) {
        Batch decode_batch = Batch::Single(
            tokens.back(), static_cast<int>(tokens.size() - 1));
        stage_start = RequestClock::now();
        logits = ForwardBatch(ctx, model, decode_batch);
        double decode_ms = ElapsedMs(stage_start);
        request.decode_ms += decode_ms;
        request.RecordEvent("decode", decode_ms, 1,
                            "pos=" + std::to_string(tokens.size() - 1));
        ++ctx.n_decode_tokens;
        ++request.decode_tokens;
        if (!dump_logits_dir.empty()) {
          DumpLogits(logits, dump_logits_dir + "/logits_step" +
                                 std::to_string(step) + ".bin");
          ++step;
        }
        stage_start = RequestClock::now();
        next_token = sampler.Sample(logits);
        request.sample_ms += ElapsedMs(stage_start);
        tokens.push_back(next_token);
        if (next_token == tokenizer->eos_id()) {
          break;
        }
      }
    } else {
      std::cout << "decode loop skipped.\n";
    }
  } catch (const std::exception& e) {
    request.SetError("Inference failed: " + std::string(e.what()));
    request.Finish();
    PrintRequestTrace(request, std::cerr);
    std::cerr << request.error << "\n";
    return 1;
  }

  if (!dump_logits_dir.empty()) {
    std::vector<int> generated(tokens.begin() + prompt_len, tokens.end());
    try {
      DumpGeneratedTokens(generated,
                          dump_logits_dir + "/generation_tokens.txt");
    } catch (const std::exception& e) {
      request.SetError("Logits dump failed: " + std::string(e.what()));
      request.Finish();
      PrintRequestTrace(request, std::cerr);
      std::cerr << request.error << "\n";
      return 1;
    }
  }

  std::vector<int> generated(tokens.begin() + prompt_len, tokens.end());
  request.generated_tokens = static_cast<int>(generated.size());
  request.RecordEvent("sample", request.sample_ms, request.generated_tokens,
                      "generated_tokens");
  request.Finish();
  std::cout << "\ngenerated tokens: [";
  for (size_t i = 0; i < generated.size(); ++i) {
    if (i > 0) {
      std::cout << ", ";
    }
    std::cout << generated[i];
  }
  std::cout << "]\n";

  std::string generated_text = tokenizer->Decode(generated);
  std::cout << "generated text: \"" << generated_text << "\"\n";
  if (backend_config.kind == BackendKind::kCuda) {
    std::cout << "\n";
    PrintCudaRuntimeSummary(model);
  }
  PrintRequestTrace(request);

  return 0;
}

// ---------------------------------------------------------------------------
// Run mode (interactive chat)
// ---------------------------------------------------------------------------
static void PrintRunUsage(const char* prog) {
  std::cout
      << "Usage: " << prog << " run <model-path|dir> [options]\n"
      << "Options:\n"
      << "  --temperature <T>  Sampling temperature (default: 0.0 = greedy)\n"
      << "  --top-k <k>        Top-k sampling (default: 0 = disabled)\n"
      << "  --seed <S>         Random seed (default: 0 = random)\n"
      << "  -n, --n-predict <n> Maximum response tokens per turn (default: "
         "64)\n"
      << "  --tokenizer <path> Path to vocab.json tokenizer file\n"
      << "  --backend cpu|cuda Execution backend (default: cpu; cuda requires "
         "-DMINI_LLAMA_CUDA=ON)\n"
      << "  --device <n>       CUDA device id for --backend cuda (default: 0)\n"
      << "  -h, --help         Show this help\n";
}

static int RunChat(int argc, char** argv) {
  if (argc >= 3 && (std::strcmp(argv[2], "-h") == 0 ||
                    std::strcmp(argv[2], "--help") == 0)) {
    PrintRunUsage(argv[0]);
    return 0;
  }

  if (argc < 3) {
    std::cerr << "Missing model directory.\n";
    PrintRunUsage(argv[0]);
    return 1;
  }

  std::string model_dir = argv[2];
  float temperature = 0.0f;
  int top_k = 0;
  unsigned int seed = 0;
  int max_response_tokens = 64;
  std::string tokenizer_path;
  BackendConfig backend_config;

  for (int i = 3; i < argc; ++i) {
    if (std::strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
      if (!ParseFloatArg(argv[++i], temperature)) {
        std::cerr
            << "Invalid --temperature value. Expected a non-negative float.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
      if (!ParseIntArg(argv[++i], top_k)) {
        std::cerr
            << "Invalid --top-k value. Expected a non-negative integer.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
      if (!ParseUintArg(argv[++i], seed)) {
        std::cerr << "Invalid --seed value. Expected a non-negative integer.\n";
        return 1;
      }
    } else if ((std::strcmp(argv[i], "-n") == 0 ||
                std::strcmp(argv[i], "--n-predict") == 0) &&
               i + 1 < argc) {
      if (!ParseIntArg(argv[++i], max_response_tokens) ||
          max_response_tokens <= 0) {
        std::cerr
            << "Invalid --n-predict value. Expected a positive integer.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
      tokenizer_path = argv[++i];
    } else if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
      if (!ParseBackendArg(argv[++i], backend_config)) {
        std::cerr << "Invalid --backend value. Supported values: cpu, cuda.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
      if (!ParseDeviceArg(argv[++i], backend_config)) {
        std::cerr
            << "Invalid --device value. Expected a non-negative integer.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "-h") == 0 ||
               std::strcmp(argv[i], "--help") == 0) {
      PrintRunUsage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown argument: " << argv[i] << "\n";
      PrintRunUsage(argv[0]);
      return 1;
    }
  }

  if (!ValidateBackendOrPrint(backend_config)) {
    return 1;
  }

  // Load model
  LoadedModel lm = LoadModelAndTokenizer(model_dir, "", tokenizer_path);
  if (!lm.model.loaded) {
    std::cerr << "Failed to load model: " << lm.model.load_error << "\n";
    return 1;
  }
  if (!lm.tokenizer) {
    std::cerr << "Failed to load tokenizer.\n";
    return 1;
  }
  if (lm.model.config.vocab_size < lm.tokenizer->vocab_size()) {
    std::cerr << "Model vocab_size must be at least "
              << lm.tokenizer->vocab_size() << " for the tokenizer.\n";
    return 1;
  }

  MiniLlamaModel& model = lm.model;
  if (!PrepareCudaWeightsOrPrint(model, backend_config)) {
    return 1;
  }
  std::unique_ptr<Tokenizer>& tokenizer = lm.tokenizer;

  PromptBuilder builder;
  if (!lm.chat_template.empty()) {
    builder.SetChatTemplate(lm.chat_template);
  }
  Terminal term;
  ChatSession session;
  session.sampler_params_.temperature = temperature;
  session.sampler_params_.top_k = top_k;
  session.sampler_params_.seed = seed;

  // Plain text mode keeps the default system message in session state.
  if (lm.chat_template.empty()) {
    session.AddMsg({"system", "You are a helpful assistant."});
  }

  term.PrintMessage("mini-llama.cpp chat");
  term.PrintMessage("backend: " + BackendKindName(backend_config.kind));
  if (backend_config.kind == BackendKind::kCuda) {
    term.PrintMessage("cuda: " + CudaDeviceSummary(backend_config.device_id));
    term.PrintMessage("uploaded weights: " +
                      std::to_string(ModelCudaUploadedWeightCount(model)));
    term.PrintMessage(
        "gpu memory used: " + std::to_string(ModelCudaMemoryBytes(model)) +
        " bytes (" + FormatMb(ModelCudaMemoryBytes(model)) + ")");
  }
  term.PrintMessage(BackendNoteText(backend_config));
  term.PrintMessage("Type /help for commands, /exit to quit.\n");
  if (lm.chat_template.empty() && model.config.max_seq_len <= 256) {
    term.PrintMessage(
        "Tiny teaching model: random weights, small context window, smoke-test "
        "output.");
    term.PrintMessage(
        "Real chat demo: ./build/mini-llama run models/chat -n 8\n");
  }

  Context ctx(&model);

  while (true) {
    term.PrintUserPrompt();
    std::string input = term.ReadLine();
    if (input.empty() && std::cin.eof()) {
      break;
    }

    // Handle commands
    if (input == "/help") {
      term.PrintHelp();
      continue;
    }
    if (input == "/exit") {
      break;
    }
    if (input == "/clear") {
      session.Clear();
      ctx = Context(&model);
      if (lm.chat_template.empty()) {
        session.AddMsg({"system", "You are a helpful assistant."});
      }
      term.PrintMessage("Chat history cleared.\n");
      continue;
    }
    if (input == "/stats") {
      term.PrintStats(session);
      continue;
    }
    if (input == "/params") {
      term.PrintParams(session.sampler_params_);
      continue;
    }
    if (!input.empty() && input[0] == '/') {
      term.PrintMessage("Unknown command: " + input + "\n");
      continue;
    }
    if (input.empty()) {
      continue;
    }

    RequestContext request =
        StartRequest("run", BackendKindName(backend_config.kind), model_dir);

    std::vector<ChatMsg> candidate_messages = session.msg_;
    candidate_messages.push_back({"user", input});

    auto stage_start = RequestClock::now();
    std::string prompt_text = builder.Build(candidate_messages);
    request.RecordEvent("prompt_build", ElapsedMs(stage_start), 0,
                        "messages=" +
                            std::to_string(candidate_messages.size()));
    stage_start = RequestClock::now();
    std::vector<int> tokens = tokenizer->Encode(prompt_text);
    request.tokenize_ms = ElapsedMs(stage_start);
    request.prompt_tokens = static_cast<int>(tokens.size());
    request.RecordEvent("tokenize", request.tokenize_ms, request.prompt_tokens,
                        "prompt");

    if (tokens.size() >= static_cast<size_t>(model.config.max_seq_len)) {
      request.SetError("prompt uses " + std::to_string(tokens.size()) +
                       " tokens, context window is " +
                       std::to_string(model.config.max_seq_len) +
                       ". Use /clear or a shorter prompt.");
      request.Finish();
      PrintRequestTrace(request);
      term.PrintMessage("Error: " + request.error + "\n");
      continue;
    }

    int max_response =
        model.config.max_seq_len - static_cast<int>(tokens.size());
    if (max_response > max_response_tokens) {
      max_response = max_response_tokens;
    }

    session.msg_ = candidate_messages;
    Sampler sampler(session.sampler_params_);
    Tensor logits;

    auto start = std::chrono::steady_clock::now();

    size_t cached_prefix_len = session.LongestCachedPrefix(tokens);
    size_t context_prefix_len = CommonPrefixLength(ctx.token_history, tokens);
    size_t prefix_len = std::min(cached_prefix_len, context_prefix_len);
    if (prefix_len >= tokens.size()) {
      ctx = Context(&model);
      prefix_len = 0;
    } else if (prefix_len == 0) {
      ctx = Context(&model);
    } else if (prefix_len < ctx.token_history.size()) {
      ctx.token_history.resize(prefix_len);
      ctx.pos = static_cast<int>(prefix_len - 1);
    }
    std::vector<int> new_prompt_tokens(
        tokens.begin() + static_cast<std::ptrdiff_t>(prefix_len), tokens.end());
    session.SetTokenHistory(tokens);

    // Prefill only the context suffix that is missing from KV cache.
    {
      Batch prefill = Batch::Prefill(new_prompt_tokens,
                                                static_cast<int>(prefix_len));
      stage_start = RequestClock::now();
      logits = ForwardBatch(ctx, model, prefill);
      request.prefill_ms = ElapsedMs(stage_start);
      request.prefill_tokens = static_cast<int>(new_prompt_tokens.size());
      request.RecordEvent(
          "prefill", request.prefill_ms, request.prefill_tokens,
          "radix_hit=" + std::to_string(cached_prefix_len) +
              ", prefix_reuse=" + std::to_string(prefix_len));
      ctx.n_prefill_tokens += static_cast<int>(new_prompt_tokens.size());
    }

    // Generate response
    std::vector<int> generated_ids;
    std::string streamed_reply;
    int generated_count = 0;

    term.PrintAssistantPrefix();
    try {
      for (int i = 0; i < max_response; ++i) {
        stage_start = RequestClock::now();
        int next_token = sampler.Sample(logits);
        request.sample_ms += ElapsedMs(stage_start);
        tokens.push_back(next_token);
        session.AppendToken(next_token);
        ++generated_count;

        if (next_token != tokenizer->eos_id()) {
          generated_ids.push_back(next_token);
          std::string current_reply = tokenizer->Decode(generated_ids);
          if (current_reply.size() > streamed_reply.size()) {
            term.PrintTokenText(current_reply.substr(streamed_reply.size()));
            term.Flush();
            streamed_reply = current_reply;
          }
        }

        Batch decode_batch =
            Batch::Single(next_token, static_cast<int>(tokens.size() - 1));
        stage_start = RequestClock::now();
        logits = ForwardBatch(ctx, model, decode_batch);
        double decode_ms = ElapsedMs(stage_start);
        request.decode_ms += decode_ms;
        request.RecordEvent("decode", decode_ms, 1,
                            "pos=" + std::to_string(tokens.size() - 1));
        ++ctx.n_decode_tokens;
        ++request.decode_tokens;
        if (next_token == tokenizer->eos_id()) {
          break;
        }
      }
    } catch (const std::exception& e) {
      term.NewLine();
      request.SetError("Inference error: " + std::string(e.what()));
      request.Finish();
      PrintRequestTrace(request);
      term.PrintMessage(request.error + "\n");
      continue;
    }

    // Decode and output
    std::string assistant_reply = tokenizer->Decode(generated_ids);
    if (assistant_reply.size() > streamed_reply.size()) {
      term.PrintTokenText(assistant_reply.substr(streamed_reply.size()));
    }
    term.NewLine();
    term.NewLine();

    auto end = std::chrono::steady_clock::now();
    double elapsed_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    session.AddMsg({"assistant", assistant_reply});
    session.RecordTurn(static_cast<int>(new_prompt_tokens.size()),
                       generated_count, elapsed_ms);
    request.generated_tokens = generated_count;
    request.RecordEvent("sample", request.sample_ms, request.generated_tokens,
                        "generated_tokens");
    session.RecordPrefix(session.token_history_);
    request.Finish();
    PrintRequestTrace(request);
  }

  term.PrintMessage("Goodbye.\n");
  return 0;
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
static void PrintGlobalUsage(const char* prog) {
  std::cout << "Usage: " << prog << " <command> [options]\n"
            << "Commands:\n"
            << "  generate      One-shot text generation (default)\n"
            << "  run           Interactive chat mode\n"
            << "  inspect       Inspect model metadata\n"
            << "  inspect-gguf  Inspect GGUF file\n"
            << "  bench         Benchmark inference performance\n"
            << "  --cuda-info   Print CUDA device information\n"
            << "\n"
            << "Run \"" << prog << " generate --help\", \"" << prog
            << " run --help\", or \"" << prog
            << " bench --help\" for details.\n";
}

static int RunCudaInfo() {
  try {
    int count = CudaDeviceCount();
    std::cout << "CUDA devices: " << count << "\n";
    for (int device_id = 0; device_id < count; ++device_id) {
      CudaDeviceInfo info = CudaGetDeviceInfo(device_id);
      double total_gb = static_cast<double>(info.total_memory_bytes) /
                        (1024.0 * 1024.0 * 1024.0);
      std::cout << "device " << info.id << ": " << info.name << "\n"
                << "  compute capability: " << info.compute_major << "."
                << info.compute_minor << "\n"
                << "  total memory: " << info.total_memory_bytes << " bytes ("
                << std::fixed << std::setprecision(2) << total_gb << " GB)\n"
                << "  driver version: " << info.driver_version << "\n"
                << "  runtime version: " << info.runtime_version << "\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "CUDA info failed: " << e.what() << "\n";
    return 1;
  }
}

// ---------------------------------------------------------------------------
// Inspect mode
// ---------------------------------------------------------------------------
static void PrintInspectUsage(const char* prog) {
  std::cout << "Usage: " << prog << " inspect <model-path|dir> [options]\n"
            << "Options:\n"
            << "  --backend cpu|cuda   Execution backend (default: cpu; cuda "
               "requires -DMINI_LLAMA_CUDA=ON)\n"
            << "  --device <n>         CUDA device id for --backend cuda "
               "(default: 0)\n"
            << "  -h, --help           Show this help\n";
}

static MiniLlamaModel LoadModelForInspect(const std::string& path) {
  std::string model_path = path;
  if (std::filesystem::is_directory(path)) {
    std::filesystem::path bin_path = std::filesystem::path(path) / "model.bin";
    std::filesystem::path json_path =
        std::filesystem::path(path) / "model.json";
    if (std::filesystem::exists(bin_path) &&
        std::filesystem::exists(json_path)) {
      return LoadModel(json_path.string(), bin_path.string());
    }
    std::string gguf_path = FindGgufInDirectory(path);
    if (!gguf_path.empty()) {
      model_path = gguf_path;
    } else {
      return LoadModel(json_path.string(), bin_path.string());
    }
  }

  if (IsGgufFile(model_path)) {
    return LoadGgufModel(model_path);
  }

  std::filesystem::path model_file(model_path);
  std::string config_path = (model_file.parent_path() / "model.json").string();
  return LoadModel(config_path, model_path);
}

static int RunInspect(int argc, char** argv) {
  if (argc >= 3 && (std::strcmp(argv[2], "-h") == 0 ||
                    std::strcmp(argv[2], "--help") == 0)) {
    PrintInspectUsage(argv[0]);
    return 0;
  }

  if (argc < 3) {
    std::cerr << "Missing model path.\n";
    PrintInspectUsage(argv[0]);
    return 1;
  }

  std::string model_path = argv[2];
  BackendConfig backend_config;
  for (int i = 3; i < argc; ++i) {
    if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
      if (!ParseBackendArg(argv[++i], backend_config)) {
        std::cerr << "Invalid --backend value. Supported values: cpu, cuda.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
      if (!ParseDeviceArg(argv[++i], backend_config)) {
        std::cerr
            << "Invalid --device value. Expected a non-negative integer.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "-h") == 0 ||
               std::strcmp(argv[i], "--help") == 0) {
      PrintInspectUsage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown argument: " << argv[i] << "\n";
      PrintInspectUsage(argv[0]);
      return 1;
    }
  }

  if (!ValidateBackendOrPrint(backend_config)) {
    return 1;
  }

  if (std::filesystem::is_directory(model_path)) {
    std::filesystem::path config_path =
        std::filesystem::path(model_path) / "model.json";
    if (std::filesystem::exists(config_path) &&
        !LLaMA_Tensor::InspectModel(config_path.string())) {
      return 1;
    }
  }

  MiniLlamaModel model = LoadModelForInspect(model_path);
  if (!model.loaded) {
    std::cerr << "Failed to load model: " << model.load_error << "\n";
    return 1;
  }

  std::cout << "backend: " << BackendKindName(backend_config.kind) << "\n";
  if (backend_config.kind == BackendKind::kCuda) {
    std::cout << "cuda: " << CudaDeviceSummary(backend_config.device_id)
              << "\n";
    if (!PrepareCudaWeightsOrPrint(model, backend_config)) {
      return 1;
    }
    PrintCudaExecutionSummary(model);
  }
  std::cout << BackendNoteText(backend_config) << "\n";

  return 0;
}

static int RunInspectGguf(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " inspect-gguf <path>\n";
    return 1;
  }
  GgufReader reader;
  if (!reader.Load(argv[2])) {
    std::cerr << "Failed to load GGUF: " << reader.load_error << "\n";
    return 1;
  }
  InspectGguf(reader);
  return 0;
}

// ---------------------------------------------------------------------------
// Bench mode
// ---------------------------------------------------------------------------
static void PrintBenchUsage(const char* prog) {
  std::cout
      << "Usage: " << prog << " bench <model-path|dir> [options]\n"
      << "Options:\n"
      << "  -p, --prompt <str>    Input prompt text (default: \"hello\")\n"
      << "  -n, --n-predict <n>   Number of tokens to generate (default: 64)\n"
      << "  --seed <S>            Random seed (default: 0 = random)\n"
      << "  --tokenizer <path>    Path to vocab.json tokenizer file\n"
      << "  --quant q8_0|q4_0     Quantize loaded Linear weights before "
         "benchmark\n"
      << "  --threads <n>         Number of threads for parallel ops (0 = "
         "auto)\n"
      << "  --q80-mode <m>        LinearQ80 dispatch path: tiled|per-element|"
         "scalar (default: tiled)\n"
      << "  --warmup <n>          Warmup iterations before timing (default: 0)\n"
      << "  --repeat <n>          Measured iterations; median is reported "
         "(default: 1)\n"
      << "  --backend cpu|cuda    Execution backend (default: cpu; cuda "
         "requires -DMINI_LLAMA_CUDA=ON)\n"
      << "  --device <n>          CUDA device id for --backend cuda (default: "
         "0)\n"
      << "  --verbose             Print debug dumps after each step\n"
      << "  -h, --help            Show this help\n";
}

static double BenchmarkRepeated(int warmup, int iterations,
                                const std::function<void()>& fn) {
  for (int i = 0; i < warmup; ++i) {
    fn();
  }

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) {
    fn();
  }
  auto end = std::chrono::steady_clock::now();
  double elapsed_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  return elapsed_ms / static_cast<double>(iterations);
}

// 取样本中位数（偶数个样本时取中间两个的均值）。
static double MedianOf(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const size_t n = values.size();
  if (n % 2 == 1) {
    return values[n / 2];
  }
  return 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

static int RunBench(int argc, char** argv) {
  if (argc >= 3 && (std::strcmp(argv[2], "-h") == 0 ||
                    std::strcmp(argv[2], "--help") == 0)) {
    PrintBenchUsage(argv[0]);
    return 0;
  }

  if (argc < 3) {
    std::cerr << "Missing model directory.\n";
    PrintBenchUsage(argv[0]);
    return 1;
  }

  std::string model_dir = argv[2];
  std::string prompt = "hello";
  int n_predict = 64;
  unsigned int seed = 0;
  std::string tokenizer_path;
  bool verbose = false;
  std::string quant_type;
  int n_threads = 0;
  std::string q80_mode = "tiled";
  int warmup = 0;
  int repeat = 1;
  BackendConfig backend_config;

  for (int i = 3; i < argc; ++i) {
    if ((std::strcmp(argv[i], "-p") == 0 ||
         std::strcmp(argv[i], "--prompt") == 0) &&
        i + 1 < argc) {
      prompt = argv[++i];
    } else if ((std::strcmp(argv[i], "-n") == 0 ||
                std::strcmp(argv[i], "--n-predict") == 0) &&
               i + 1 < argc) {
      if (!ParseIntArg(argv[++i], n_predict)) {
        std::cerr << "Invalid --n-predict value.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
      if (!ParseUintArg(argv[++i], seed)) {
        std::cerr << "Invalid --seed value.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
      tokenizer_path = argv[++i];
    } else if (std::strcmp(argv[i], "--quant") == 0 && i + 1 < argc) {
      quant_type = argv[++i];
    } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      if (!ParseIntArg(argv[++i], n_threads) || n_threads < 0) {
        std::cerr << "Invalid --threads value.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--q80-mode") == 0 && i + 1 < argc) {
      q80_mode = argv[++i];
    } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
      if (!ParseIntArg(argv[++i], warmup)) {
        std::cerr << "Invalid --warmup value.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) {
      if (!ParseIntArg(argv[++i], repeat) || repeat < 1) {
        std::cerr << "Invalid --repeat value (must be >= 1).\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
      if (!ParseBackendArg(argv[++i], backend_config)) {
        std::cerr << "Invalid --backend value. Supported values: cpu, cuda.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
      if (!ParseDeviceArg(argv[++i], backend_config)) {
        std::cerr
            << "Invalid --device value. Expected a non-negative integer.\n";
        return 1;
      }
    } else if (std::strcmp(argv[i], "--verbose") == 0) {
      verbose = true;
    } else if (std::strcmp(argv[i], "-h") == 0 ||
               std::strcmp(argv[i], "--help") == 0) {
      PrintBenchUsage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown argument: " << argv[i] << "\n";
      PrintBenchUsage(argv[0]);
      return 1;
    }
  }

  if (!quant_type.empty() && quant_type != "q8_0" && quant_type != "q4_0") {
    std::cerr << "Invalid --quant value: " << quant_type
              << ". Supported values: q8_0, q4_0.\n";
    return 1;
  }

  LinearQ80Mode q80_path = DefaultLinearQ80Mode();
  if (!ParseLinearQ80Mode(q80_mode, q80_path)) {
    std::cerr << "Invalid --q80-mode value: " << q80_mode
              << ". Supported values: tiled, per-element, scalar.\n";
    return 1;
  }

  if (!ValidateBackendOrPrint(backend_config)) {
    return 1;
  }

  LoadedModel lm = LoadModelAndTokenizer(model_dir, "", tokenizer_path);
  if (!lm.model.loaded) {
    std::cerr << "Failed to load model: " << lm.model.load_error << "\n";
    return 1;
  }
  if (!lm.tokenizer) {
    std::cerr << "Failed to load tokenizer.\n";
    return 1;
  }
  if (lm.model.config.vocab_size < lm.tokenizer->vocab_size()) {
    std::cerr << "Model vocab_size must be at least "
              << lm.tokenizer->vocab_size() << " for the tokenizer.\n";
    return 1;
  }

  MiniLlamaModel baseline_model = lm.model;
  MiniLlamaModel& model = lm.model;
  std::unique_ptr<Tokenizer>& tokenizer = lm.tokenizer;
  std::vector<int> tokens = tokenizer->Encode(prompt);
  if (tokens.size() > static_cast<size_t>(model.config.max_seq_len)) {
    std::cerr << "Prompt too long.\n";
    return 1;
  }
  if (tokens.size() + static_cast<size_t>(n_predict) >
      static_cast<size_t>(model.config.max_seq_len)) {
    n_predict = model.config.max_seq_len - static_cast<int>(tokens.size());
  }

  try {
    ApplyQuantOverride(model, quant_type);
  } catch (const std::exception& e) {
    std::cerr << "Quantization failed: " << e.what() << "\n";
    return 1;
  }
  if (!PrepareCudaWeightsOrPrint(model, backend_config)) {
    return 1;
  }
  std::cout << "Benchmark: " << model_dir << "\n";
  std::cout << "  backend: " << BackendKindName(backend_config.kind) << "\n";
  if (backend_config.kind == BackendKind::kCuda) {
    PrintCudaExecutionSummary(model, "  ");
  }
  std::cout << "  " << BackendNoteText(backend_config) << "\n";
  std::cout << "  prompt: \"" << prompt << "\" (" << tokens.size()
            << " tokens)\n";
  SetThreadCount(n_threads);
  SetLinearQ80Mode(q80_path);
  std::cout << "  n_predict: " << n_predict << "\n";
  std::cout << "  seed: " << seed << "\n";
  std::cout << "  quant: " << (quant_type.empty() ? "model-native" : quant_type)
            << "\n";
  std::cout << "  threads: " << GetThreadCount() << "\n";
  std::cout << "  q80-mode: " << LinearQ80ModeName(q80_path) << "\n";
  std::cout << "  warmup/repeat: " << warmup << "/" << repeat << "\n";
  std::cout << "  verbose: " << (verbose ? "true" : "false") << "\n\n";

  // 预热 + 正式测量。每次 RunBenchmark 都新建 Context/KV cache，彼此独立；
  // 进程内重复执行可避免重复加载 1.4GB 模型，统计口径为正式的 repeat 次
  // 取中位数。
  for (int i = 0; i < warmup; ++i) {
    RunBenchmark(model, tokens, n_predict, seed, verbose);
  }
  std::vector<BenchmarkResult> samples;
  samples.reserve(static_cast<size_t>(repeat));
  for (int i = 0; i < repeat; ++i) {
    samples.push_back(RunBenchmark(model, tokens, n_predict, seed, verbose));
  }

  std::vector<double> prefill_samples;
  std::vector<double> decode_samples;
  prefill_samples.reserve(samples.size());
  decode_samples.reserve(samples.size());
  for (const BenchmarkResult& sample : samples) {
    prefill_samples.push_back(sample.prefill_ms);
    decode_samples.push_back(sample.decode_ms);
  }
  BenchmarkResult result = samples.back();
  result.prefill_ms = MedianOf(prefill_samples);
  result.decode_ms = MedianOf(decode_samples);

  std::cout << "Results";
  if (repeat > 1) {
    std::cout << " (median of " << repeat << " runs";
    if (warmup > 0) {
      std::cout << ", " << warmup << " warmup discarded";
    }
    std::cout << ")";
  }
  std::cout << ":\n";
  if (repeat > 1) {
    std::cout << "  prefill ms samples: ";
    for (size_t i = 0; i < prefill_samples.size(); ++i) {
      if (i > 0) std::cout << ", ";
      std::cout << std::fixed << std::setprecision(2) << prefill_samples[i];
    }
    std::cout << "\n";
    std::cout << "  decode ms samples:  ";
    for (size_t i = 0; i < decode_samples.size(); ++i) {
      if (i > 0) std::cout << ", ";
      std::cout << std::fixed << std::setprecision(2) << decode_samples[i];
    }
    std::cout << "\n";
  }
  std::cout << "  prompt tokens:     " << result.n_prompt_tokens << "\n";
  std::cout << "  generated tokens:  " << result.n_generated_tokens << "\n";
  std::cout << "  Decode tokens:     " << result.n_decode_tokens << "\n";
  std::cout << "  prefill time:      " << std::fixed << std::setprecision(2)
            << result.prefill_ms << " ms\n";
  std::cout << "  Decode time:       " << std::fixed << std::setprecision(2)
            << result.decode_ms << " ms\n";
  std::cout << "  total time:        " << std::fixed << std::setprecision(2)
            << (result.prefill_ms + result.decode_ms) << " ms\n";
  std::cout << "  tokens/s (total):  " << std::fixed << std::setprecision(2)
            << result.tokens_per_sec() << "\n";
  std::cout << "  tokens/s (Decode): " << std::fixed << std::setprecision(2)
            << result.decode_tokens_per_sec() << "\n";
  if (backend_config.kind == BackendKind::kCuda) {
    PrintCudaRuntimeSummary(model, "  ");
  }

  // Memory footprint
  size_t actual_bytes = ModelWeightBytes(model);
  size_t f32_bytes = ModelWeightBytesF32(model);
  std::cout << "\n  weight memory:\n";
  std::cout << "    actual:    " << actual_bytes << " bytes (" << std::fixed
            << std::setprecision(2) << (actual_bytes / (1024.0 * 1024.0))
            << " MB)\n";
  std::cout << "    f32 equiv: " << f32_bytes << " bytes (" << std::fixed
            << std::setprecision(2) << (f32_bytes / (1024.0 * 1024.0))
            << " MB)\n";
  std::cout << "    savings:   " << std::fixed << std::setprecision(2)
            << (static_cast<double>(f32_bytes) / actual_bytes)
            << "x compression\n";

  if (!quant_type.empty()) {
    try {
      Tensor baseline_logits = RunLogitsForTokens(baseline_model, tokens);
      Tensor quant_logits = RunLogitsForTokens(model, tokens);
      auto [max_err, mean_err] = LogitsError(baseline_logits, quant_logits);

      std::cout << "\n  logits error vs model-native:\n";
      std::cout << "    max:  " << std::scientific << std::setprecision(3)
                << max_err << "\n";
      std::cout << "    mean: " << std::scientific << std::setprecision(3)
                << mean_err << "\n";
    } catch (const std::exception& e) {
      std::cerr << "Failed to compute logits error: " << e.what() << "\n";
      return 1;
    }
  }

  return 0;
}

}  // namespace mini_llama

int main(int argc, char** argv) {
  if (argc < 2) {
    LLaMA_Tensor::PrintGlobalUsage(argv[0]);
    return 1;
  }

  std::string command = argv[1];
  if (command == "generate") {
    return LLaMA_Tensor::RunGenerate(argc, argv);
  } else if (command == "run") {
    return LLaMA_Tensor::RunChat(argc, argv);
  } else if (command == "inspect") {
    return LLaMA_Tensor::RunInspect(argc, argv);
  } else if (command == "bench") {
    return LLaMA_Tensor::RunBench(argc, argv);
  } else if (command == "inspect-gguf") {
    return LLaMA_Tensor::RunInspectGguf(argc, argv);
  } else if (command == "--cuda-info") {
    return LLaMA_Tensor::RunCudaInfo();
  } else if (command == "-h" || command == "--help") {
    LLaMA_Tensor::PrintGlobalUsage(argv[0]);
    return 0;
  } else if (!command.empty() && command[0] == '-') {
    // Backward compatibility: default to generate mode when first arg is a
    // flag. Shift argv[1..] down by inserting "generate" at position 1.
    std::vector<char*> shifted_argv;
    shifted_argv.reserve(argc + 1);
    shifted_argv.push_back(argv[0]);
    shifted_argv.push_back(const_cast<char*>("generate"));
    for (int i = 1; i < argc; ++i) {
      shifted_argv.push_back(argv[i]);
    }
    return LLaMA_Tensor::RunGenerate(argc + 1, shifted_argv.data());
  } else {
    std::cerr << "Unknown command: " << command << "\n";
    LLaMA_Tensor::PrintGlobalUsage(argv[0]);
    return 1;
  }
}
