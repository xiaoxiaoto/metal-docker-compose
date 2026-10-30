# MOSS-Transcribe-Diarize · vLLM + vLLM Metal


## 共享优化接入（默认关闭）

本目录使用基础vLLM Metal显式provider合同；扩展功能分包、兼容及验收规则见根 `AGENTS.md`。
启动时固化公共 `METAL_MODEL_OPTIMIZERS`、`METAL_QUANTIZATION`、
`METAL_QUANTIZATION_PROFILE`、`METAL_ANE_PROFILE`、`METAL_STREAMING_PROFILE`；
默认量化 off、provider 列表为空。dynamic 须显式选择 `dynamic_quantization` 并提供绝对路径批准 profile；
ANE 和 streaming 同样须显式选择各自 provider 并提供绝对路径 profile。没有完整真实任务批准时保持 off。

**当前阻塞：** 私有准备器要求 `vllm-metal==0.31.0` 和 `mlx-audio` 0.5.8 或 0.5.9，而共享锁使用 Metal 0.32.0，版本门禁会拒绝准备。须验证并更新 STT 适配兼容性后才能部署；不要绕过或放宽门禁。

参考 [官方模型卡](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize) 与 `../higgs-tts-3/`。**不使用 vLLM-Omni，不自建 HTTP server。**
根目录将本模型配置为 **dynamic，空闲300秒卸载**。每次根目录启动（包括默认/gateway/lifecycle）先安全停止所有依赖模型、删除历史共享环境和私有 overlay，再重建共享后端；停止失败不删除，构建失败不回退。`./start.sh moss` 每次调用本目录 `./start.sh --prepare-only` 重建私有环境和 plist，首次推理才离线加载。权重、下载缓存及日志保留；参数以本次设置为准。详见[根目录重建合同](../../README.md#每次根目录启动前重建共享环境)。

冷模型不在聚合模型列表中，单模型身份查询返回503且不唤醒；先查看根目录 `python3 -I runtime_mgmt/lifecycle_control.py status`，发送下文真实转写curl，推理完成后复查身份。冷请求另留最多600秒启动预算；不要旁路8097推理，否则无法获得请求保活保证。


## 为什么官方 vLLM 能运行，Metal 还需要适配？

vLLM 已实现 `MossTranscribeDiarizeForConditionalGeneration`：Whisper 音频 encoder → adaptor → Qwen3 decoder，以及音频 processor、转写 prompt 和 `diarized_json` 输出解析。官方 CUDA 路径直接使用这套实现。

vLLM Metal 的 STT 路径则通过 MLX 执行模型，不直接运行 CUDA/PyTorch 模型类；当前上游 STT registry 只有 Whisper / Qwen3-ASR，不能因为 vLLM 注册了 MOSS 就直接加载它。另一方面，已安装的 `mlx-audio 0.5.8` 有完整 MOSS 模型实现，因此本目录只补上缺失的桥接：

- `moss_metal.py`：进程内注册 MOSS STT constructor；保留 vLLM processor 产生的 mel、每 chunk 有效长度和 chunk 数量；复用 mlx-audio 的 encoder / adaptor / decoder，逐 chunk 编码。
- 复用原生 vLLM Metal `STTModelRunner`，不复制 scheduler、模型或服务器。请求级采样走 `STTSampling`，每次请求新建 decode cache；不伪造达到 token 预算后的 EOS。
- Whisper / Qwen3-ASR 的原有请求提取路径保持不变。权重强制严格加载，缺失 tensor、量化权重、无效特征和 placeholder 数量不匹配都会拒绝。

**这是本地适配插件，不是上游已经原生支持 MOSS 的承诺。** 它使用 Metal 的私有 STT hooks；依赖改变时必须重新验证。

## 启停

需要 Apple Silicon、macOS 15+、Homebrew、Python 3.11+（根控制脚本与启动引导）、可用的 `modelscope` CLI；根目录总代理另需 Docker Compose。模型实际使用 uv 托管的 Python `>=3.12,<3.13` 共享运行时。共享依赖仅由根目录 [`pyproject.toml`](../../pyproject.toml) / [`uv.lock`](../../uv.lock) 声明，以 `uv sync --locked` 非 editable 安装：vLLM/Metal `0.32.0`，Omni 固定 `9842ebc17cf1d9c6b440ecb6a79039162f68a1dd`，Omni Metal 来自 `https://github.com/xiaoxiaoto/vllm-omni-metal.git` 的 `5774df7d720b3a8c6d32035729b42c31e480f346`。共享准备不依赖相邻源码，保留官方联网预检但不随 main 自动升级，MPS 与 `pip check` 等验证通过后才激活；详见[根目录合同](../../README.md)。

vLLM 与 vllm-metal 也从官方 GitHub 源码安装，分别固定 `vllm-project/vllm` 的 `5df63a7203cdb9328d824a27b682e77c7b9928be` 与 `vllm-project/vllm-metal` 的 `6d6f2c37f8242f1ff1a336534adf6c2d2fe2936d`（v0.32.0）；由 uv 隔离构建非 editable wheel，不使用官方预编译 wheel。vLLM 的 CPU 编译形态只是 Metal 插件底座，不代表模型 CPU fallback。

只准备、不立即加载：

```bash
cd composes/moss-transcribe-diarize
./start.sh --prepare-only
```

`--prepare-only` 执行依赖检查、环境准备、版本检查、模型下载/验证和 plist 生成，通过 `plutil -lint` 后退出，不 bootstrap 或加载模型。已有任务或 8097 冲突仍拒绝；未提供 `VLLM_BACKEND_PYTHON` 时会调用共享准备器，安全停用全部依赖模型并删除历史环境后重建。非法参数非零退出；无参数仍准备后立即加载：

```bash
cd composes/moss-transcribe-diarize
./start.sh
curl -fsS http://127.0.0.1:8097/health
# 根目录 ./start.sh 启动总代理后：
curl -fsS 'http://127.0.0.1/v1/models?model=moss-transcribe-diarize'
./stop.sh
```

- 模型在 macOS launchctl 中监听 `127.0.0.1:8097`；统一入口由根目录总代理提供：`http://127.0.0.1/v1/`，按 `model` 分流。本目录不再包含 Docker/Nginx 代理。Linux 容器不能使用 Apple Metal。
- 根目录 dynamic 入口每次重建共享后端并重新准备本模型环境/plist，不立即加载模型。直接无参数运行本目录脚本在未提供 `VLLM_BACKEND_PYTHON` 时也调用共享准备器。兼容组以根目录 README 和 `common_env/check_versions.py` 为准；MOSS 请求仍只使用 vLLM STT，不进入 Omni pipeline。
- 私有环境位于 `~/.local/share/local-model-backend/overlays/moss-transcribe-diarize/<fingerprint>/`，只安装本目录插件，通过 `.pth` 复用 `current` 指向的共享后端；不改 Homebrew 或统一环境的模型实现。共享基础后端位于同一根目录的 `runtime/<fingerprint>/`。
- 新建 overlay 不运行 ensurepip；pip 和基础依赖通过明确的共享 base `.pth` 继承。准备器变化会产生新指纹，不复用旧私有 pip 环境；原有版本门禁和 `pip check` 保持不变。
- 安装校验使用选定私有Python `-I`。MOSS走基础vllm-metal的MLX STT，不使用
  本插件执行层，不要求其量化API；与PaddleOCR/H2O等基础vLLM路由一样，
  不属于本次Omni DynamicEngine接入范围。不能用Omni入口接通冒充MOSS真实量化批准。
- 插件限制为 `VLLM_PLUGINS=metal,moss_stt`，模型进程离线运行。使用单请求、8192 上下文、BF16，关闭 async scheduling、chunked prefill 和 prefix caching。
- 本目录启停脚本只管理 MOSS，不要求 Docker，也不停止总代理。根目录重建共享后端时按公共策略安全停止所有依赖模型；有在途请求或停止失败时拒绝继续，不删除旧环境。
- launchctl 工作目录位于 Application Support，避免 Documents 的 macOS 隐私权限问题。总代理配置和启停见根目录 `README.md`。

### 模型准备

MOSS 仍扫描 `~/.cache/modelscope/hub/models/作者/模型/`：一个完整的 MOSS 兼容模型自动选择，多个时输入编号，没有时输入魔搭仓库名（作者/模型），通过 `modelscope download --revision master --local_dir` 下载并校验，不默认从 Hugging Face 下载。这与 `../h2o-lightning-4b/` 当前采用的 Hugging Face 官方手动下载流程不同。

已有完整缓存不会重新下载或更新；无效缓存不进入候选列表，也不会被下载覆盖。模型 ID 优先从 `.mdl` 中只解析字符串，不反序列化执行；无元数据时使用目录名。输入关闭会明确报错，需在终端运行。原始未量化权重约 **1.82 GB**；这不是运行时内存上限，也不是最低 Mac 内存承诺。

选中模型卡仓库名 `/` 后的完整文本转为全小写，同步用于 `--served-model-name`、健康检查和请求的 `model` 字段，例如 `openmoss/MOSS-Transcribe-Diarize` → `moss-transcribe-diarize`；不带作者前缀或 `:latest`。下载仓库名和缓存路径不变；更换模型时须同步根目录路由及身份检查。

显式指定本地模型：

```bash
MOSS_MODEL_DIR=/absolute/path/to/model ./start.sh
```

显式目录沿用部署名称 `moss-transcribe-diarize`。会校验配置、processor、tokenizer、权重索引和分片 tensor；无效目录直接失败，不下载替代模型。HF snapshot 到本仓库 blobs 的正常符号链接可用；越界路径拒绝。vLLM 的 `--trust-remote-code` 用于官方自定义 config / processor，所以只能使用信任的本地 checkpoint；不要指向未知仓库代码。

## 转写与说话人分离

将下例音频路径替换为实际存在的 WAV 文件，先使用短音频。

```bash
set -o pipefail
curl --fail-with-body --show-error --connect-timeout 5 --max-time 10 \
  'http://127.0.0.1/v1/models'

curl --fail-with-body --show-error --connect-timeout 5 --max-time 10 \
  --get 'http://127.0.0.1/v1/models' \
  --data-urlencode 'model=moss-transcribe-diarize'

curl --fail-with-body --show-error --connect-timeout 5 --max-time 1800 \
  'http://127.0.0.1/v1/audio/transcriptions' \
  --form 'model=moss-transcribe-diarize' \
  --form 'file=@/实际路径/音频.wav;type=audio/wav' \
  --form 'response_format=diarized_json' \
  --form 'temperature=0' \
  --form 'max_completion_tokens=2048' | \
  python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin), ensure_ascii=False, indent=2))'
```

`diarized_json` 返回 `text` 和带 `start/end/speaker` 的 `segments`。字段/错误语义由 vLLM 提供；不是 Higgs `/v1/audio/speech`，也不使用 SGLang 的 `max_new_tokens` 字段。仅要文本可选 `response_format=json`。

请求体上限 25 MiB，等待 1800 秒。音频 placeholder、prompt 和输出共享 8192 上下文；从短音频开始，长音频可能超出上下文或截断输出，须检查完整性。该路径是 one-shot STT：没有逐 token 的调度/KV 分页收益，不承诺即时取消、流式延迟或生产并发。不能用单说话人样本推断多人分离准确率。

接口无鉴权，仅绑定本机；不要改成公网监听。

## 检查与日志

```bash
backend="$(python3 ../../common_env/prepare_backend.py)"
base="$backend/bin/python"
runtime="$("$base" prepare_runtime.py)"
bash -n start.sh stop.sh
docker compose -f ../../docker-compose.yml config --quiet
```

绕过代理单独诊断原生服务（前台，可用 Ctrl-C 停止）：

```bash
model_info="$("$runtime/bin/python" prepare.py)"
model_name="${model_info%%$'\n'*}"
model="${model_info#*$'\n'}"
VLLM_PLUGINS=metal,moss_stt HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$runtime/bin/vllm" serve "$model" \
  --served-model-name "$model_name" \
  --trust-remote-code --host 127.0.0.1 --port 8097 \
  --dtype bfloat16 --max-model-len 8192 --max-num-seqs 1 \
  --enforce-eager --no-enable-prefix-caching --no-enable-chunked-prefill --no-async-scheduling
```

此时将请求 URL 改为 `http://127.0.0.1:8097/v1/audio/transcriptions`。该诊断绕过生命周期保活，不应用于正常请求。

- 模型日志：`~/Library/Logs/moss-transcribe-diarize-metal.log`
- 错误日志：`~/Library/Logs/moss-transcribe-diarize-metal.error.log`
- 代理日志：在根目录执行 `docker compose logs --tail 100 gateway`

官方模型权重为 Apache-2.0；依赖和本地部署不改变各自许可。
