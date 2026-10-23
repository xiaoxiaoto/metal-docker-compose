# MOSS-Transcribe-Diarize · vLLM + vLLM Metal

参考 [官方模型卡](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize) 与 `../higgs-tts-3/`。**不使用 vLLM-Omni，不自建 HTTP server。**
根目录将本模型配置为 **dynamic，空闲300秒卸载**，详见[通用生命周期](../README.md#通用模型生命周期)。根目录 `./start.sh moss` 在缺少 plist 时自动调用本目录 `./start.sh --prepare-only`，完成环境、模型和 plist 准备；已有 plist 则校验后复用，不改写模型精度。两种情况都不加载模型，首次推理才触发离线加载，无需先手动启动再停止。已手动运行的服务不会被管理器接管或停止。

冷模型不在聚合模型列表中，单模型身份查询返回503且不唤醒；先查看根目录 `python3 -I lifecycle_control.py status`，发送下文真实转写curl，推理完成后复查身份。冷请求另留最多600秒启动预算；不要旁路8097推理，否则无法获得请求保活保证。


## 为什么官方 vLLM 能运行，Metal 还需要适配？

vLLM 已实现 `MossTranscribeDiarizeForConditionalGeneration`：Whisper 音频 encoder → adaptor → Qwen3 decoder，以及音频 processor、转写 prompt 和 `diarized_json` 输出解析。官方 CUDA 路径直接使用这套实现。

vLLM Metal 的 STT 路径则通过 MLX 执行模型，不直接运行 CUDA/PyTorch 模型类；当前上游 STT registry 只有 Whisper / Qwen3-ASR，不能因为 vLLM 注册了 MOSS 就直接加载它。另一方面，已安装的 `mlx-audio 0.5.8` 有完整 MOSS 模型实现，因此本目录只补上缺失的桥接：

- `moss_metal.py`：进程内注册 MOSS STT constructor；保留 vLLM processor 产生的 mel、每 chunk 有效长度和 chunk 数量；复用 mlx-audio 的 encoder / adaptor / decoder，逐 chunk 编码。
- 复用原生 vLLM Metal `STTModelRunner`，不复制 scheduler、模型或服务器。请求级采样走 `STTSampling`，每次请求新建 decode cache；不伪造达到 token 预算后的 EOS。
- Whisper / Qwen3-ASR 的原有请求提取路径保持不变。权重强制严格加载，缺失 tensor、量化权重、无效特征和 placeholder 数量不匹配都会拒绝。

**这是本地适配插件，不是上游已经原生支持 MOSS 的承诺。** 它使用 Metal 的私有 STT hooks；依赖改变时必须重新验证。

## 启停

需要 Apple Silicon、macOS 15+、Homebrew、Python 3.9+（准备）、可用的 `modelscope` CLI；根目录总代理另需 Docker Compose。模型实际使用统一隔离后端的 Python 3.12。

只准备、不立即加载：

```bash
cd docker-compose/moss-transcribe-diarize
./start.sh --prepare-only
```

`--prepare-only` 完整复用依赖检查、共享与私有环境准备、版本检查、模型下载/验证和 plist 生成，写入并通过 `plutil -lint` 后成功退出，不执行 `launchctl bootstrap`，不加载或先启动再停止所选模型。准备模式仍拒绝已有注册任务和 8097 端口冲突，不接管或停止当前所选服务；共享后端如需升级，仍沿用原流程，可能停止其他受影响的本部署模型，应先确认无在途请求。未知参数、重复参数或多余参数均非零退出。无参数行为不变，仍准备后立即加载：

```bash
cd docker-compose/moss-transcribe-diarize
./start.sh
curl -fsS http://127.0.0.1:8097/health
# 根目录 ./start.sh 启动总代理后：
curl -fsS 'http://127.0.0.1/v1/models?model=moss-transcribe-diarize'
./stop.sh
```

- 模型在 macOS launchctl 中监听 `127.0.0.1:8097`；统一入口由根目录总代理提供：`http://127.0.0.1/v1/`，按 `model` 分流。本目录不再包含 Docker/Nginx 代理。Linux 容器不能使用 Apple Metal。
- 根目录 dynamic 入口缺少 plist 时先准备后登记，已有 plist 只校验复用；不会立即加载模型。直接无参数运行本目录脚本仍先查询官方最新版本、升级并校验统一后端，再加载模型。四服务共用 vLLM/Metal `0.31.0` 与配套 Omni/Omni Metal 开发版，详见根目录 `README.md`；MOSS 请求仍只使用 vLLM STT，不进入 Omni pipeline。
- 私有环境位于 `~/.local/share/local-model-backend/overlays/moss-transcribe-diarize/<fingerprint>/`，只安装本目录插件，通过 `.pth` 复用 `current` 指向的共享后端；不改 Homebrew 或统一环境的模型实现。共享基础后端位于同一根目录的 `runtime/<fingerprint>/`。
- 安装校验使用选定私有Python `-I`。MOSS走基础vllm-metal的MLX STT，不使用
  本插件执行层，不要求其量化API；与PaddleOCR/H2O等基础vLLM路由一样，
  不属于本次Omni DynamicEngine接入范围。不能用Omni入口接通冒充MOSS真实量化批准。
- 原有真实 HTTP 验收使用 `0.30.0`，现在部署仅接受统一 `0.31.0` 核心。mlx-audio 接受 `0.5.8` / `0.5.9`，已核对两版 MOSS 模型目录源码相同；其他版本拒绝启动，升级后仍须真实转写验收。
- 插件限制为 `VLLM_PLUGINS=metal,moss_stt`，模型进程离线运行。使用单请求、8192 上下文、BF16，关闭 async scheduling、chunked prefill 和 prefix caching。
- 本目录启停脚本只管理 MOSS，不要求 Docker，也不停止总代理。若共享后端版本/源码变化，根目录预检按公共策略停止所有已注册的本部署模型后再切换；这会中断在途请求。
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

```bash
curl --fail-with-body --max-time 1800 \
  http://127.0.0.1/v1/audio/transcriptions \
  -F model=moss-transcribe-diarize \
  -F file=@audio.wav \
  -F response_format=diarized_json \
  -F temperature=0 \
  -F max_completion_tokens=2048
```

`diarized_json` 返回 `text` 和带 `start/end/speaker` 的 `segments`。字段/错误语义由 vLLM 提供；不是 Higgs `/v1/audio/speech`，也不使用 SGLang 的 `max_new_tokens` 字段。仅要文本可选 `response_format=json`。

请求体上限 25 MiB，等待 1800 秒。音频 placeholder、prompt 和输出共享 8192 上下文；从短音频开始，长音频可能超出上下文或截断输出，须检查完整性。该路径是 one-shot STT：没有逐 token 的调度/KV 分页收益，不承诺即时取消、流式延迟或生产并发。不能用单说话人样本推断多人分离准确率。

接口无鉴权，仅绑定本机；不要改成公网监听。

## 检查与日志

```bash
backend="$(python3 ../prepare_backend.py)"
base="$backend/bin/python"
runtime="$("$base" prepare_runtime.py)"
bash -n start.sh stop.sh
docker compose -f ../docker-compose.yml config --quiet
```

本次真实官方权重 HTTP 验收：6.28 秒中文 WAV，返回正确文本、`S01` 和 `0.07–6.23` 秒 segment，约 1.1 秒响应；重复请求结果一致。**不是生产性能基准或 ASR/diarization 全集评测。**

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

此时将请求 URL 改为 `http://127.0.0.1:8097/v1/audio/transcriptions`。测试未升级 Homebrew 或修改 Higgs/H2O/YuE2；验收后卸载本次模型任务。

- 模型日志：`~/Library/Logs/moss-transcribe-diarize-metal.log`
- 错误日志：`~/Library/Logs/moss-transcribe-diarize-metal.error.log`
- 代理日志：在根目录执行 `docker compose logs --tail 100 gateway`

官方模型权重为 Apache-2.0；依赖和本地部署不改变各自许可。
