# mini-llama

一个用 C++17 实现的 LLaMA / Qwen2 推理引擎，支持 **CPU** 与 **NVIDIA CUDA** 两种执行后端。项目包含 GGUF 模型加载、Tokenizer、KV Cache、批处理前向、采样器、交互式终端，以及可复现的性能基准工具。

> 当前工作区基于 `feature/cuda_runtime` 分支（HEAD `d9c6de9`）。`main` 分支不包含本文描述的完整 CUDA 接入。
>
> 本文中的性能数据来自 2026-10-22 的历史基准报告，具体来源和适用范围的说明见 [基准数据说明](#基准数据说明)。

## 功能特性

| 能力 | CPU 后端 | CUDA 后端 |
|---|---|---|
| F32 / Q8_0 / Q4_0 / Q4_1 Linear | 支持 | 支持 |
| RMSNorm / RoPE / SiLU / SwiGLU | CPU 算子 | CUDA kernel |
| Attention | CPU + CPU KV Cache | CUDA attention + GPU KV Cache |
| 批量 prefill | 支持 | 支持连续 position 的整批 prefill |
| 单 token decode | 支持 | 设备端前向 |
| Sampler | CPU | CPU（logits 回传后采样） |
| 运行时统计 | — | Linear / Activation / Attention 调用数、H2D/D2H、KV 读写 |
| 量化格式 | F32 / Q8_0 / Q4_0 / Q4_1 | F32 / Q8_0 / Q4_0 / Q4_1 |

CUDA 路径当前的重点是：

- 模型权重一次性上传到 GPU，decode 期间不重复上传。
- QKV、attention、WO、FFN、final norm、lm_head 尽量保持 device-resident。
- GPU KV Cache 支持写入、读取和按层清零。
- `bench` 会输出 CUDA 调用计数和 fallback 计数，便于确认是否真的在 GPU 上执行。

## 仓库结构

```text
mini-llama/
├── CMakeLists.txt
├── README.md
├── include/            # 头文件：模型、算子、KV Cache、CUDA、CLI 相关类型
├── src/                # C++ / CUDA 实现
│   ├── main.cpp        # CLI 入口
│   ├── forward.cpp     # CPU / CUDA 前向
│   ├── model.cpp       # 模型加载与 CUDA 权重上传
│   ├── backend.cpp     # CPU / CUDA 后端选择与校验
│   ├── cuda_*.cpp/.cu  # CUDA runtime、matmul、attention、quant、tensor
│   └── ...
├── models/
│   └── chat/           # 放置 Qwen2 GGUF 模型
└── build*/             # 本地构建目录（不提交）
```

当前仓库没有额外的 benchmark 编排脚本或原始 JSONL 数据；本文所有复现命令都只依赖 `mini-llama` 自身的 CLI。

## 环境要求

| 组件 | 要求 |
|---|---|
| 操作系统 | Ubuntu 22.04 / 24.04（其他 Linux 也可，需自行调整依赖安装） |
| 编译器 | 支持 C++17 的 g++ / clang++ |
| CMake | ≥ 3.17（CUDA 构建建议 ≥ 3.18，以支持 `CMAKE_CUDA_ARCHITECTURES`） |
| simdjson | CPU 和 CUDA 构建都需要 |
| CUDA（可选） | CUDA Toolkit + NVIDIA 驱动；仅 CUDA 构建需要 |
| GPU（可选） | NVIDIA GPU，计算能力建议 ≥ 8.0 |
| 模型 | Qwen2-0.5B-Instruct Q8_0 GGUF（或其他受支持的 GGUF） |

安装 CPU 依赖：

```bash
sudo apt update
sudo apt install -y build-essential cmake libsimdjson-dev
```

如果使用 conda 环境，也可以安装 conda-forge 的 simdjson：

```bash
conda install -c conda-forge simdjson
export CMAKE_PREFIX_PATH="$CONDA_PREFIX:$CMAKE_PREFIX_PATH"
```

`simdjson` 不在默认搜索路径时，可在配置阶段显式指定：

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -Dsimdjson_DIR="$CONDA_PREFIX/lib/cmake/simdjson"
```

## 快速开始：CPU

### 1. 获取代码

```bash
git clone -b feature/cuda_runtime \
  https://github.com/cpptrainingcamp/mini-llama.git
cd mini-llama
```

如果该分支已经合并到默认分支，可以省略 `-b feature/cuda_runtime`。

### 2. 构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
```

如需复现 CPU 报告中的 SIMD 构建，可额外加入：

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-mavx2 -mfma"
cmake --build build -j"$(nproc)"
```

> 默认 `Release` 使用 `-O2`。`-mavx2 -mfma` 是报告机器的构建参数，不是所有 CPU 都支持；请先确认自己的 CPU 指令集。

### 3. 准备模型

```bash
mkdir -p models/chat

# 示例：从 Hugging Face 下载 Qwen2-0.5B-Instruct 的 Q8_0 GGUF
# 文件名以官方仓库页面为准；下载后放到 models/chat/
huggingface-cli download Qwen/Qwen2-0.5B-Instruct-GGUF \
  --include "*q8_0.gguf" --local-dir models/chat

ls -lh models/chat
```

> 如果 `huggingface-cli` 不存在，先执行 `pip install -U huggingface_hub`。

`--model` 既可以指向单个 `.gguf` 文件，也可以指向包含 GGUF 的目录。GGUF 自带 tokenizer 时不需要额外指定 `--tokenizer`。

### 4. 运行

```bash
# 单次生成
./build/mini-llama generate \
  --model models/chat \
  -p "你好" \
  -n 32 \
  --backend cpu

# 交互聊天（模型路径是位置参数）
./build/mini-llama run models/chat --backend cpu -n 64

# 查看模型配置和 tensor metadata
./build/mini-llama inspect models/chat

# 查看 GGUF metadata
./build/mini-llama inspect-gguf \
  models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf
```

## CUDA 构建与运行

### 1. 确认 GPU 和 CUDA Toolkit

```bash
nvidia-smi
nvcc --version
```

如果 `nvcc` 不在 `PATH` 中，配置时显式指定：

```bash
cmake -S . -B build-cuda \
  -DMINI_LLAMA_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DCMAKE_CUDA_ARCHITECTURES=89
```

`CMAKE_CUDA_ARCHITECTURES=89` 对应 RTX 40 系列 / Ada（SM 8.9）；其他 GPU 请改成对应架构，例如 A100 用 `80`，H100 用 `90`。不指定时 CMake 默认也是 `89`。

### 2. 构建

```bash
cmake -S . -B build-cuda \
  -DMINI_LLAMA_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89

cmake --build build-cuda -j"$(nproc)"
```

### 3. 检查设备

```bash
./build-cuda/mini-llama --cuda-info
```

预期输出类似：

```text
CUDA devices: 1
device 0: NVIDIA GeForce RTX 4090 D
  compute capability: 8.9
  total memory: 25280839680 bytes (23.54 GB)
  driver version: 13030
  runtime version: 13010
```

### 4. GPU 推理

```bash
./build-cuda/mini-llama generate \
  --model models/chat \
  -p "你好" \
  -n 32 \
  --backend cuda \
  --device 0

./build-cuda/mini-llama run models/chat \
  --backend cuda --device 0 -n 64
```

CUDA 模式会先上传模型权重，再执行 prefill / decode。`bench` 输出中的以下字段可用于确认执行路径：

- `uploaded weights`：上传到 GPU 的权重数量
- `gpu memory used`：GPU 权重占用
- `cuda Linear calls`：CUDA Linear 调用次数
- `cuda activation calls`：CUDA 激活 / 逐元素调用次数
- `cuda attention calls`：CUDA attention 调用次数
- `cpu attention fallback calls`：回退到 CPU attention 的次数
- `host->device copies` / `device->host copies`：运行期传输次数

> `--device` 只能与 `--backend cuda` 一起使用。CPU 构建执行 `--backend cuda` 会明确报错，而不是静默回退。

## 常用命令

| 场景 | 命令 |
|---|---|
| CPU 单次生成 | `./build/mini-llama generate --model models/chat -p "你好" -n 32 --backend cpu` |
| CUDA 单次生成 | `./build-cuda/mini-llama generate --model models/chat -p "你好" -n 32 --backend cuda --device 0` |
| 交互聊天 | `./build-cuda/mini-llama run models/chat --backend cuda --device 0 -n 64` |
| 查看模型 | `./build/mini-llama inspect models/chat` |
| 查看 GGUF | `./build/mini-llama inspect-gguf models/chat/Qwen2-0.5B-Instruct-Q8_0.gguf` |
| CUDA 设备信息 | `./build-cuda/mini-llama --cuda-info` |
| CPU 基准 | `./build/mini-llama bench models/chat -p hello -n 32 --q80-mode tiled --warmup 2 --repeat 5 --threads 14` |
| CUDA 基准 | `./build-cuda/mini-llama bench models/chat --backend cuda --device 0 -p hello -n 32 --warmup 2 --repeat 5` |

### 主要参数

| 参数 | 适用命令 | 说明 |
|---|---|---|
| `--model <path>` | `generate` | 模型 GGUF 文件或包含 GGUF 的目录 |
| `--config <path>` | `generate` | 仅用于仓库自带的极小配置模型 |
| `-p, --prompt <text>` | `generate` / `bench` | 输入 prompt |
| `-n, --n-predict <N>` | `generate` / `run` / `bench` | 最多生成 N 个 token |
| `--temperature <T>` | `generate` / `run` | 采样温度，`0` 表示贪心 |
| `--top-k <K>` | `generate` / `run` | top-k 采样 |
| `--seed <N>` | `generate` / `run` / `bench` | 随机种子 |
| `--tokenizer <path>` | `generate` / `run` / `bench` | 仅在使用独立 tokenizer 文件时需要 |
| `--quant <type>` | `generate` / `bench` | 加载后将 Linear 重新量化为 `q8_0` / `q4_0` |
| `--threads <N>` | `generate` / `bench` | CPU 线程数，默认 `0`（自动使用硬件并发；报告使用 14） |
| `--backend <cpu\|cuda>` | `generate` / `run` / `inspect` / `bench` | 选择后端，默认 `cpu` |
| `--device <id>` | 使用 CUDA 后端的命令 | CUDA 设备编号，仅 CUDA 后端有效 |
| `--q80-mode <mode>` | `bench` | Q8_0 路径：`tiled` / `per-element` / `scalar` |
| `--warmup <N>` | `bench` | 预热轮数 |
| `--repeat <N>` | `bench` | 正式测量轮数 |
| `--verbose` | `bench` | 输出更详细的加载 / 执行信息 |
| `--dump-logits <dir>` | `generate` | 将每一步 logits 导出到目录，用于调试和数值对比 |

不同子命令支持的参数略有差异；以 `./build/mini-llama <command> --help` 的输出为准。

查看完整帮助：

```bash
./build/mini-llama --help
./build-cuda/mini-llama --cuda-info
```

## 基准测试口径

本文的两个基准报告使用相同的核心口径：

- **预热**：正式计时前运行 2 次。
- **正式测量**：重复 5 次，取中位数。
- **排除加载时间**：模型加载、Tokenizer 加载、CUDA 权重上传不计入单次请求耗时。
- **prefill**：处理完整 prompt 的阶段；报告用 `prefill time` 近似 TTFT。
- **decode**：逐 token 生成阶段；`decode tok/s` 按生成的 token 数除以 decode 总时间计算。
- **total tok/s**：报告中的端到端吞吐列，按报告原值抄录；其分子 / 口径在不同报告中不完全一致，跨行比较请优先使用 `prefill` 和 `decode` 两列。
- **随机性**：CPU 报告使用贪心采样（`--temperature 0`）；CUDA 报告固定 `--seed 42` 并使用报告中的默认采样参数。

不同机器、不同驱动、不同编译参数下的绝对数字不可直接比较；同一台机器上的相对差异更有参考价值。

## CPU 性能

### 测试环境

| 项目 | 配置 |
|---|---|
| CPU | Intel Core Ultra 5 225H |
| 核心 / 线程 | 14C / 14T |
| 指令集 | AVX2 + FMA |
| 虚拟化 | Hyper-V 虚拟机 |
| 操作系统 | Linux |
| 编译器 | g++ 15.2.0 |
| CMake | 4.2.3 |
| 构建 | Release，`-O2 -mavx2 -mfma` |
| 模型 | Qwen2-0.5B-Instruct-Q8_0.gguf |
| 采样 | 贪心（`--temperature 0`） |
| 计时 | 2 次预热 + 5 次正式测量，取中位数 |

模型规模：

| 项目 | 值 |
|---|---:|
| 参数量 | 0.5B |
| 词表大小 | 151936 |
| hidden dim | 896 |
| 层数 | 24 |
| attention heads | 14 |
| KV heads | 2 |
| head dim | 64 |
| FFN hidden dim | 4864 |
| 最大上下文 | 32768 |
| 权重实际占用（原生 Q8_0 + F32 lm_head） | 1401.48 MiB |
| 若全部转为 F32 | 约 2403.90 MiB |

### 1. Q8_0 Linear 路径对比

固定 `prompt=1`、`decode=64`、14 线程，只改变 `--q80-mode`：

| 路径 | prefill ms | prefill ms/token | decode tok/s | decode ms/token | total tok/s | 相对 scalar |
|---|---:|---:|---:|---:|---:|---:|
| scalar | 563.1 | 563.1 | 1.82 | 548.8 | 1.82 | 1.00× |
| per-element | 367.8 | 367.8 | 2.61 | 382.2 | 2.61 | 1.43× |
| tiled | 362.5 | 362.5 | 2.67 | 374.1 | 2.68 | 1.47× |

结论：

- `per-element` 相对 `scalar` 的 decode 吞吐提升 **+43.4%**，单 token 延迟降低 **30.3%**。
- `tiled` 相对 `scalar` 的 decode 吞吐提升 **+46.7%**，单 token 延迟降低 **31.8%**。
- `tiled` 比 `per-element` 再快约 **2.3%**，是当前 Q8_0 的推荐路径。
- `--q80-mode` 只影响 Q8_0 的 `LinearQ80`；tiled 同时改善 prefill 和 decode，但 decode 是耗时更长、收益更直接的热点。

### 2. 线程扩展

固定 `prompt=1`、`decode=64`、`--q80-mode tiled`：

| 线程数 | prefill ms | decode tok/s | total tok/s | 相对 1 线程 | 并行效率 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1248.6 | 2.03 | 2.03 | 1.00× | 100.0% |
| 2 | 691.9 | 2.48 | 2.48 | 1.22× | 61.1% |
| 4 | 475.0 | 2.72 | 2.72 | 1.34× | 33.5% |
| 8 | 404.7 | 2.73 | 2.73 | 1.34× | 16.8% |
| 14 | 362.5 | 2.75 | 2.75 | 1.35× | 9.7% |

结论：

- 从 1 线程到 4 线程，decode 吞吐从 2.03 提升到 2.72 tok/s。
- 4 线程之后基本饱和，8 / 14 线程只带来约 1% 的额外收益。
- 14 线程相对 1 线程仅 **1.35×**，并行效率 **9.7%**；这是 0.5B Q8_0 模型内存带宽受限的表现。
- 实际部署建议从 4 线程起步，再根据机器负载微调。

### 3. 多负载矩阵

固定 14 线程、`--q80-mode tiled`；`prompt` 和 `decode` 为 token 数：

| prompt | decode | prefill ms | prefill ms/token | decode tok/s | decode ms/token | total tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 370.4 | 370.4 | 2.72 | 367.7 | 2.60 |
| 1 | 32 | 371.0 | 371.0 | 2.70 | 370.0 | 2.61 |
| 1 | 64 | 362.5 | 362.5 | 2.67 | 374.1 | 2.68 |
| 32 | 16 | 717.9 | 22.43 | 2.64 | 369.8 | 2.61 |
| 32 | 32 | 716.5 | 22.39 | 2.64 | 370.0 | 2.61 |
| 32 | 64 | 729.3 | 22.79 | 2.59 | 378.0 | 2.58 |
| 64 | 16 | 1358.0 | 21.22 | 2.65 | 367.5 | 1.96 |
| 64 | 32 | 1279.6 | 19.99 | 2.64 | 369.1 | 2.00 |
| 64 | 64 | 1357.8 | 21.22 | 2.59 | 377.3 | 1.99 |

> `total tok/s` 按报告原值抄录；长 prompt 行无法用表中 `prefill + decode` 精确复算，引用时以 `prefill` 和 `decode` 两列为准。

批量 prefill 的结论（来自报告）：

- prompt=32 时，prefill 摊薄到约 **22.4 ms/token**。
- prompt=64 时，prefill 摊薄到约 **21.1 ms/token**。
- 与 prompt=1 的约 350–370 ms/token 相比，批量 prefill 把每 token 成本降低了一个数量级。
- 这说明当前 CPU 实现的主要优化空间在 decode 的逐 token 矩阵乘，而不是长 prompt 的 prefill。

### 4. 与历史基线的一致性

报告将本次结果与同一台机器上的历史基线做了交叉检查：

| 指标 | 历史基线 | 本次报告 | 变化 |
|---|---:|---:|---:|
| prompt=1 / decode=16 cold prefill | 684.0 ms | 376.2 ms | -45.0% |
| prompt=1 / decode=16 cold decode | 2.52 tok/s | 2.73 tok/s | +8.1% |
| prompt=1 / decode=64 warm decode | 2.67 tok/s | 2.67 tok/s | 0.0% |

结论：decode 路径与历史暖样本一致；prefill 的差异主要来自热态 / 冷态和页缓存，而不是代码回归。

## CUDA 性能

### 测试环境

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 D |
| 显存 | 24 GB |
| 计算能力 | SM 8.9（Ada） |
| 驱动 | 610.43.03 |
| CUDA Runtime | 13.1（runtime 13010） |
| 操作系统 | Ubuntu 22.04.5 |
| 模型 | Qwen2-0.5B-Instruct-Q8_0.gguf |
| 后端 | CUDA，原生 GPU 权重（F32 lm_head + Q8_0 其余线性层） |
| 上传权重 | 1401.48 MiB |
| 采样 | 固定 `--seed 42` |
| 计时 | 2 次预热 + 5 次正式测量，取中位数 |

模型配置与 CPU 部分相同：24 层、hidden 896、14 heads、2 KV heads、head dim 64、FFN 4864、词表 151936。

### 1. GPU 稳态矩阵

9 个 prompt / decode 组合，数值全部来自报告：

| prompt | decode | prefill ms | prefill ms/token | decode ms/token | decode tok/s | total tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 17.62 | 17.62 | 16.92 | 59.09 | 58.94 |
| 1 | 32 | 17.19 | 17.19 | 17.33 | 57.71 | 57.72 |
| 1 | 64 | 17.68 | 17.68 | 19.04 | 52.53 | 52.59 |
| 32 | 16 | 545.43 | 17.04 | 17.49 | 57.18 | 19.81 |
| 32 | 32 | 555.42 | 17.36 | 17.26 | 57.95 | 29.35 |
| 32 | 64 | 539.71 | 16.87 | 18.04 | 55.44 | 38.19 |
| 64 | 16 | 1133.56 | 17.71 | 17.57 | 56.90 | 11.45 |
| 64 | 32 | 1129.44 | 17.65 | 17.28 | 57.87 | 19.22 |
| 64 | 64 | 1106.53 | 17.29 | 18.16 | 55.08 | 28.44 |

读法：

- **decode 稳态**：无论 prompt 多长，decode 大约维持在 **52.5–59.1 tok/s**，单 token 约 **16.9–19.0 ms**。
- **prefill 单 token 成本**：都在 **16.9–17.7 ms/token**，与 decode 接近。
- **瓶颈不是 HBM 带宽**：原生模型每 token 读取约 1.47 GB 权重，17 ms 对应有效带宽约 **77–87 GB/s**，远低于 HBM 峰值；耗时主要来自每 token **434 次 kernel 启动 / 调度**和小算子开销。
- **端到端吞吐**：prompt 越长，prefill 在总时间中的占比越高，`total tok/s` 越低；p64/n16 只有 11.45 tok/s。
- **最优 decode 吞吐**：p1/n16 的 59.09 tok/s；**最优长 prompt 端到端**：p32/n64 的 38.19 tok/s。

> 报告中的原生模型使用 F32 `lm_head`，因此在旧实现中批量 prefill 不会启用，prompt 会逐 token 串行处理；这是 p32/p64 的 prefill 时间接近 `prompt × 17 ms` 的原因。当前 `mini-llama` 的 CUDA 门控与旧报告存在实现差异，详见 [基准数据说明](#基准数据说明)。

扩展上下文（decode=256，KV Cache 随 position 增长）：

| p/n | prefill ms | decode ms/token | decode tok/s | KV read bytes |
|---:|---:|---:|---:|---:|
| 1/256 | 17.42 | 18.59 | 53.80 | 5659164672 |
| 64/256 | 1146.86 | 18.95 | 52.76 | 8780513280 |

在 319 个总 token 的上下文下，decode 仍保持约 **52.8 tok/s**；KV 读取字节增加约 55%，但没有改变“kernel 调度主导”的结论。

### 2. 冷启动 vs 预热

同一台机器、同一模型、`prompt=1`、`decode=16`：

| 测量 | prefill ms | decode tok/s | total tok/s |
|---|---:|---:|---:|
| 上一会话冷启动 | 92.30 | 53.34 | 47.51 |
| 本次冷启动 | 98.48 | 57.49 | 50.18 |
| 预热 2+5 中位数 | 17.19 | 57.71 | 57.72 |

结论：

- 冷启动 prefill 整体是稳态的 **5.4–5.7×**；本次为 98.48 ms vs 17.19 ms，主要来自 CUDA context、kernel 首次加载、权重首次触碰和页缓存未命中。
- 上一会话冷启动的 decode 比稳态慢约 **7.6%**；本次冷启动的 decode 已接近稳态，说明 decode kernel 路径本身已经比较稳定。
- 对外报告性能时应同时给出冷启动和稳态数字，不能只取预热后的最好值。

### 3. 每 token 的 CUDA 内核构成

以 24 层模型、单 token 为例：

| 阶段 | 内核调用数 | 占比 |
|---|---:|---:|
| Q8_0 Linear（QKV / WO / FFN） | 168 | 38.7% |
| F32 lm_head | 1 | 0.2% |
| Activation / elementwise（RMSNorm、RoPE、SiLU、残差等） | 241 | 55.5% |
| Attention | 24 | 5.5% |
| **合计** | **434** | **100%** |

结论：

- 单 token 要发起 **434 次 kernel 调用**，其中超过一半是 activation / elementwise 小算子。
- Q8_0 Linear 虽然计算量大，但调用数只占 38.7%；大量小 kernel 的启动和调度开销是后续优化重点。
- F32 `lm_head` 只占 0.2% 的调用数，却占 **37.1%** 的权重字节；A/B 实验估算它只贡献约 **6–9%** 的 decode 时间，进一步支持“调度受限而非带宽受限”的判断。
- 优化方向包括：算子融合（RMSNorm + Linear、SiLU + Mul、残差 + Norm）、CUDA Graph、以及减少不必要的 D2H/H2D 同步。

### 4. 计数器校验

报告中用运行时计数器验证“GPU 路径确实执行了预期的工作量”。每行检查 10 项恒等式，9 行共 **90/90 项通过**。

代表性行：

| p/n | 前向调用 | Linear calls | Activation calls | Attention calls | KV read bytes | D2H 次数 / bytes |
|---|---:|---:|---:|---:|---:|---:|
| 1/32 | 32 | 5408 | 7712 | 768 | 90832896 | 32 / 19447808 |
| 32/64 | 95 | 16055 | 22895 | 2280 | 784465920 | 95 / 57735680 |
| 64/64 | 127 | 21463 | 30607 | 3048 | 1398276096 | 127 / 77183488 |

恒等式包括：

- `Linear calls = forward calls × 169`（24 层 × 7 个线性层 + 1 个 lm_head）
- `Activation calls = forward calls × 241`
- `Attention calls = forward calls × 24`
- `KV read bytes = Σ(每次 forward 的当前 position) × 24 × 2 × heads × head_dim × sizeof(float)`
- `D2H copies = forward calls`，`D2H bytes = forward calls × vocab × sizeof(float)`
- `H2D copies = 0`（权重上传阶段不计入 forward 统计）

这些恒等式全部成立，说明 batch prefill 和 decode 的计数口径一致，没有遗漏层或重复计数。

### 5. 全 Q8 + 批量 prefill 探索

将 `lm_head` 也量化为 Q8_0 后，`CudaBatchPrefillSupported` 会放行整批 prefill。报告中的探索性结果：

| p/n | prefill ms | prefill ms/token | decode ms/token | decode tok/s | total tok/s | 相对原生 prefill 加速 |
|---|---:|---:|---:|---:|---:|---:|
| 1/64 | 16.80 | 16.80 | 17.69 | 56.52 | 56.56 | 1.05× |
| 1/32 | 16.69 | 16.69 | 17.46 | 57.28 | 57.36 | 1.03× |
| 32/64 | 56.02 | 1.75 | 17.32 | 57.75 | 55.80 | **9.63×** |
| 32/32 | 58.14–63.86 | 1.82–2.00 | 17.45–17.61 | 56.79–57.31 | 52.91–52.98 | **8.7–9.6×** |
| 64/64 | 115.43 | 1.80 | 17.18 | 58.22 | 53.44 | **9.59×** |
| 64/32 | 146.51 | 2.29 | 17.82 | 56.11 | 45.78 | **7.7×** |

结论：

- 全 Q8 批量 prefill 把 p32/p64 的 prefill 从约 **17 ms/token** 降到 **1.75–2.29 ms/token**，提升约 **8–10×**。
- decode 吞吐基本不变（55–58 tok/s），说明批量 prefill 不会拖慢后续逐 token 生成。
- p32/n64 的端到端吞吐从 38.19 提升到 55.80 tok/s，已经接近 p1/n64 的水平。
- 全 Q8 将权重从 **1401.48 MiB** 降到 **1020.11 MiB**（约 −27%），但 decode 只变化 **−3% 到 +8%**；这再次说明 decode 的瓶颈是 kernel 调度，而不是权重带宽。

**正确性警告：**

- p64/n64 的最大 logits 误差为 `2.85e-2`，平均误差 `4.73e-3`。
- p64/n32 曾出现一次 `sampler: logits must be finite` 失败，说明全 Q8 `lm_head` 在长 prompt 下存在数值不稳定。
- 因此，全 Q8 + 批量 prefill 目前只应作为**探索性能力**；生产配置建议保留原生 F32 `lm_head`，除非你愿意为速度接受数值风险并自行做回归验证。

### 6. CUDA 性能结论

1. **稳态 decode** 约 **52.5–59.1 tok/s**，单 token **16.9–19.0 ms**，是当前实现最稳定的指标。
2. **单 token 434 次 kernel 调用**，其中 55.5% 是 activation / elementwise；下一步应优先做算子融合和 CUDA Graph。
3. **冷启动 prefill** 比稳态慢 **5.4–5.7×**，benchmark 必须预热；冷启动数字适合描述用户体验，稳态数字适合描述 kernel 效率。
4. **报告中的旧原生实现长 prompt prefill 按 token 串行**，受 F32 `lm_head` 门控影响；当前 HEAD 已支持可上传 F32 权重的批量路径，需用新基准重新评估。全 Q8 的 **8–10×** 是旧报告下的探索性结果，且有数值风险。
5. 对 0.5B 模型而言，RTX 4090 D 的算力不是瓶颈，kernel 调度和小算子开销才是。

## 复现命令

以下命令只使用当前仓库已有的 CLI；报告中的矩阵采集脚本和原始 JSONL 不在 `mini-llama` 仓库中。

### CPU：路径对比

```bash
for mode in scalar per-element tiled; do
  echo "===== q80-mode=$mode ====="
  ./build/mini-llama bench models/chat \
    -p hello -n 64 \
    --q80-mode "$mode" \
    --warmup 2 --repeat 5 --threads 14
done
```

### CPU：线程扩展

```bash
for threads in 1 2 4 8 14; do
  echo "===== threads=$threads ====="
  ./build/mini-llama bench models/chat \
    -p hello -n 64 \
    --q80-mode tiled \
    --warmup 2 --repeat 5 --threads "$threads"
done
```

### CPU：workload 矩阵

```bash
P32="$(python3 -c 'print("hello " * 31 + "hello")')"
P64="$(python3 -c 'print("hello " * 63 + "hello")')"

for prompt in "$P32" "$P64"; do
  for n in 16 32 64; do
    echo "===== prompt_tokens≈32/64, n=$n ====="
    ./build/mini-llama bench models/chat \
      -p "$prompt" -n "$n" \
      --q80-mode tiled --warmup 2 --repeat 5 --threads 14
  done
done
```

`hello` 重复构造只是近似对应 32 / 64 token；实际 token 数请以 `bench` 输出的 `prompt tokens` 为准。完整矩阵耗时较长，建议先单独跑一组确认参数。

### CUDA：稳态矩阵

```bash
./build-cuda/mini-llama bench models/chat \
  --backend cuda --device 0 \
  -p hello -n 32 \
  --warmup 2 --repeat 5
```

把 `-p` 换成更长的 prompt、把 `-n` 改成 16 / 32 / 64，即可覆盖报告中的 9 个组合。实际 token 数请以 `bench` 输出的 `prompt tokens` 为准。

### CUDA：全 Q8 批量 prefill（探索性）

```bash
./build-cuda/mini-llama bench models/chat \
  --backend cuda --device 0 \
  --quant q8_0 \
  -p hello -n 32 \
  --warmup 2 --repeat 5
```

> `--quant q8_0` 会在加载后把全部 Linear（包括 `lm_head`）重新量化为 Q8_0，改变数值路径和权重内存占用。请只在模型副本或实验环境中使用，并检查 logits 是否有限、输出是否可接受。

## 基准数据说明

两部分数据均来自 **2026-10-22** 的历史报告：

| 报告 | 代码版本 | 机器 | 说明 |
|---|---|---|---|
| CPU Q8_0 基准 | commit `16d95b7`（历史仓库） | Intel Core Ultra 5 225H，14C/14T，Hyper-V | 报告记录了 scalar / per-element / tiled、线程扩展和 workload 矩阵 |
| CUDA RTX 4090 D 基准 | commit `1a9c4f82`（`mini-llama-cpp`，工作区含 CUDA 改动） | RTX 4090 D 24 GB，CUDA 13.1 | 报告记录了稳态矩阵、冷/暖对比、计数器校验和全 Q8 探索 |

需要强调：

1. 上述 commit **不在当前 `mini-llama` 仓库的 HEAD（`d9c6de9`）中**；表格描述的是该实现代际的行为。
2. CPU 报告的矩阵脚本 `scripts/bench_cpu_matrix.py` 和原始文件 `benchmarks/cpu_bench_2026-10-22.jsonl` 属于历史/关联仓库，当前仓库没有这些文件。
3. CUDA 报告中的原生模型使用 F32 `lm_head`；在旧实现中，这会关闭批量 prefill，导致长 prompt 按 token 串行。当前 `mini-llama` 的 `CudaLinearReady()` 接受可上传的 F32 权重，因此当前代码的批量 prefill 行为不同；本地 2-token smoke test 只报告 **169 次 Linear 调用**（不是 2×169），说明该 prompt 已按批处理执行。正式的长 prompt 性能仍需按新代码重跑。
4. 本文保留报告原始结论，是为了记录性能演进和优化方向，不是把旧数字当作当前 HEAD 的回归基线。需要当前数字时，请按上面的复现命令在目标机器重跑。
5. 报告机器是虚拟机 / 单卡环境，存在调度、频率和页缓存噪声；绝对数值不应与其他硬件直接比较。

## 已知限制

- **模型覆盖有限**：报告只验证了 Qwen2-0.5B-Instruct Q8_0；其他模型、量化格式和上下文长度需要单独验证。
- **量化格式有限**：当前加载器和 CUDA Linear 主要覆盖 F32 / Q8_0 / Q4_0 / Q4_1；K-quants、IQ-quants 等格式不支持。
- **采样在 CPU**：CUDA 路径只把 logits 回传到主机，top-k / temperature / 随机采样仍由 CPU 完成。
- **批量 prefill 要求连续 position**：当前 CUDA batch prefill 面向连续 token 的 prefill；带空洞的 paged / 非连续 KV 场景不在覆盖范围内。
- **无并发请求调度**：当前是单请求、单序列推理；没有 continuous batching 或多用户调度。
- **CPU 基准的 F32 路径未展开**：报告聚焦 Q8_0 的 `--q80-mode`；F32 `MatmulMode` 在代码中固定为 `kThreadedSimd`，没有对应的 CLI 开关。
- **全 Q8 `lm_head` 有数值风险**：长 prompt 下出现过非有限 logits；不要把探索性配置直接用于生产。
- **冷启动波动大**：CUDA context 初始化、kernel 首次加载、页缓存都会影响首轮结果；必须预热后再比较。
- **无内核级 profiling**：报告的 `ncu` 因 `ERR_NVGPUCTRPERM` 失败，`nsys` 未安装；434 次 kernel 和各阶段占比来自运行时计数器与 A/B 实验，不是 profiler 采样。
- **单卡单机**：没有多 GPU、多机或 NVLink 场景的数据。

## 常见问题

### `Could not find a package configuration file provided by "simdjson"`

指定 simdjson 的 CMake 配置目录：

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -Dsimdjson_DIR="$CONDA_PREFIX/lib/cmake/simdjson"
```

或安装系统包：

```bash
sudo apt install -y libsimdjson-dev
```

### `nvcc` 找不到

```bash
cmake -S . -B build-cuda \
  -DMINI_LLAMA_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DCMAKE_CUDA_ARCHITECTURES=89
```

### `--cuda-info` 提示没有 CUDA 设备

```bash
nvidia-smi
```

确认驱动已加载、容器已透传 GPU（`--gpus all`），并且当前用户有访问 `/dev/nvidia*` 的权限。

### CPU 构建下使用 `--backend cuda`

CPU 构建不包含 CUDA runtime，会直接报错。请使用 `build-cuda/mini-llama`，或在配置时加 `-DMINI_LLAMA_CUDA=ON` 重新构建。

### `--device` 报错

`--device` 只能与 `--backend cuda` 一起使用。CPU 路径请去掉该参数。

### `libcudart.so` 找不到

把 CUDA runtime 目录加入动态库搜索路径：

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:"$LD_LIBRARY_PATH"
```

如果使用自定义 CUDA 安装目录，请替换为实际的 `lib64` 路径。

### 显存不足

Qwen2-0.5B Q8_0 的原生权重约占 **1401.48 MiB**，还需要 KV Cache、activation 和 CUDA workspace。显存不足时可以：

- 关闭其他占用 GPU 的进程；
- 减小 `-n` 或上下文长度；
- 使用更小的模型 / 更低的量化位宽；
- 确认没有重复创建多个 CUDA context。

### `sampler: logits must be finite`

这通常意味着前向产生了 NaN / Inf。优先检查：

- 是否使用了实验性的全 Q8 `lm_head` + 长 prompt；
- 输入是否超过模型上下文长度；
- 权重是否正确加载、量化类型是否匹配；
- 是否混用了不同模型目录中的 GGUF / tokenizer。

## 参考输出

CPU 生成：

```text
$ ./build/mini-llama generate --model models/chat -p "你好" -n 16 --backend cpu
...
prefill: ... ms
decode: ... tok/s
```

CUDA 设备：

```text
$ ./build-cuda/mini-llama --cuda-info
CUDA devices: 1
device 0: NVIDIA GeForce RTX 4090 D
  compute capability: 8.9
  total memory: 25280839680 bytes (23.54 GB)
```

具体输出会随模型、编译参数和驱动版本变化；以本机实际输出为准。
