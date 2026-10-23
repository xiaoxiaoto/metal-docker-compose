# Higgs TTS 3 · vLLM-Omni + Omni Metal

Apple Silicon 本机使用 **vllm-omni + vllm-omni-metal**，执行上游原生 **Talker → Code2Wav** 双阶段。本目录不再运行 Docker 代理；统一入口由根目录总代理提供，不在 Linux 容器中加载 Metal 模型。

不再使用旧 `higgs_metal`、mlx-audio 融合适配器或自建 HTTP server；接口、模型采样和 codec 由上游 Omni 提供。
根目录现将本模型配置为 **dynamic，空闲300秒卸载**。通用配置和状态见[根目录生命周期说明](../README.md#通用模型生命周期)。根目录 `./start.sh higgs` 在缺少 plist 时自动调用本目录 `./start.sh --prepare-only`，完成环境、模型和 plist 准备；已有 plist 则校验后复用，不覆盖其精度或量化策略。两种情况都不加载真实模型，首次推理才触发离线加载，无需先手动启动再停止。管理器不接管已经手动启动的服务。

冷模型不会出现在聚合列表，单模型身份查询返回503且不唤醒；先用根目录 `python3 -I lifecycle_control.py status` 检查登记，再发送下文真实推理curl，完成后复查身份。冷请求在原600秒推理预算之外另留最多600秒加载时间。动态保活只覆盖经总代理的请求，勿旁路直连8095推理。


## 环境

- Apple Silicon macOS 15+、MPS、Homebrew、Python 3.9+（启动准备）、ModelScope CLI。根目录总代理另需 Docker Compose。
- `start.sh` 使用根目录 `prepare_backend.py` 准备的统一隔离环境作为基础，再构建/复用 Higgs 独占的正常 Omni wheel overlay。总启动先查询官方版本并检查兼容性；显式设置已验证的 `VLLM_BACKEND_PYTHON` 时不调用共享准备器。需要升级时先停止受影响模型，构建新环境后切换；不使用独立 Homebrew HEAD 安装/升级流程。
- 四个服务共用 vLLM/Metal `0.32.0` 与配套 Omni/Omni Metal `0.32.0.dev0`。Omni Metal 从相邻源码仓库构建非 editable wheel，包含通用输出协议修复；不依赖未发布的远端 Formula，不修改 Homebrew 或本地未提交源码。细节见根目录 `README.md`。
- `prepare.py` 自动扫描 `~/.cache/modelscope/hub/models/作者/模型/`，只提供完整且未量化的 `higgs_multimodal_qwen3` 缓存。唯一候选自动选择，多个候选编号选择，没有候选时输入 ModelScope 仓库名（如 `bosonai/higgs-audio-v3-tts-4b`），下载 `master` 后重新校验；完整缓存不重新下载。
- 部署名称取所选模型卡仓库名 `/` 后的完整文本并转为全小写，例如 `bosonai/higgs-audio-v3-tts-4b` → `higgs-audio-v3-tts-4b`；`prepare.py` 依次输出部署名称、目录，供启动和健康检查复用。显式目录有 `.mdl` 时从中读取仓库名，否则沿用 `higgs-audio-v3-tts-4b`。更换模型时须同步根目录路由及身份检查。
- `HIGGS_MODEL_DIR` 显式指定本地目录时优先使用并校验，不扫描、不下载；无交互 stdin 且需要选择时明确报错。`.mdl` 只解析字符串，不反序列化执行缓存代码。模型路径、索引及所有 safetensors 分片必须完整，量化模型及其他架构不会进入候选列表。
- 软件准备会从官方 Omni main 和本地通用后端源码构建/复用一致版本组；模型只在准备阶段下载，launchctl 进程离线加载。失败保留缓存、不启动模型。
- 根目录 `./start.sh higgs` / `all` 仅在需要准备或显式启动时传入已校验的 `VLLM_BACKEND_PYTHON`；已有动态 plist 只校验复用。直接运行本目录脚本也使用同一预检。没有变化时不重装、不升级运行中的环境。
- 后端已在 48 GiB Mac 验证；这里不是最低内存承诺。不要同时加载新旧两份服务。

`metal-deploy.yaml` 基于上游 Higgs profile（已核对固定 Omni main，非 XPU 拓扑未变），保留原生 sampling、connector 和 async chunk。两个阶段设置 `max_num_seqs=1`、`CUSTOM` MPS SDPA、同步调度和关闭 prefix caching；Code2Wav 模型专用动态量化适配只存在于 Higgs 私有 Omni wheel。每阶段 `gpu_memory_utilization=0.3`，KV 分配保留系统内存 headroom。

## 启停与迁移

只准备、不立即加载：

```bash
cd docker-compose/higgs-tts-3
./start.sh --prepare-only
```

准备模式保留磁盘检查；16GiB可用内存门槛只约束实际模型加载，HTTP冷启动仍执行生命周期资源门禁。

`--prepare-only` 复用无参数启动的依赖检查、模型发现/下载校验、共享与私有环境准备及 plist 生成，在写入并通过 `plutil -lint` 后成功退出，不执行 `launchctl bootstrap`，不加载或先启动再停止所选模型。准备模式仍拒绝已有注册任务和 8095 端口冲突，不接管或停止当前所选服务；共享后端如需升级，仍沿用原流程，可能停止其他受影响的本部署模型，应先确认无在途请求。未知参数、重复参数或多余参数均非零退出。无参数 `./start.sh` 仍按原流程准备并立即加载，以下为显式启动/迁移用法：

```bash
cd docker-compose/higgs-tts-3
# 已注册旧任务或需要重新加载配置时，必须先停止：
./stop.sh
./start.sh
curl -fsS http://127.0.0.1:8095/health
# 根目录 ./start.sh 启动总代理后：
curl -fsS 'http://127.0.0.1/v1/models?model=higgs-audio-v3-tts-4b'
```

本机模型监听 `127.0.0.1:8095`，统一入口为根目录 `http://127.0.0.1/v1/`，按请求的 `model` 路由。`start.sh` 拒绝复用已注册的旧任务或杀掉占用端口的其他进程；下次启动生成新的 launchctl plist，直接执行选定安装环境的 `bin/vllm-omni serve --omni`，不依赖源码目录或 shell 的 venv 激活状态。

launchctl label 仍为 `com.local.higgs-tts-3-metal`，配置在 `~/Library/Application Support/higgs-tts-3/`，不是登录自启项。`stop.sh` 只停止该模型任务，不操作总代理或其他模型。统一后端升级可能停止本部署其他受影响模型，但不修改 Homebrew、不停止其他项目；单独 `stop.sh` 的管理范围不变。

Omni `/health` 成功时没有旧服务的 JSON 字段；模型身份通过 `/v1/models` 检查。health 就绪不等于语音生成验收通过。

仅供本机使用，无鉴权，不要暴露公网。Nginx保留16KiB请求体限制和关闭代理缓冲；生命周期代理按配置限制请求期限，网关额外容纳冷启动等待。

## 生成语音

```bash
curl --fail-with-body --max-time 600 \
  http://127.0.0.1/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"higgs-audio-v3-tts-4b","input":"你好，欢迎使用语音合成。","response_format":"wav","stream":false,"seed":42,"max_new_tokens":512}' \
  -o /tmp/higgs-demo.wav
# 确认请求成功后播放：
afplay /tmp/higgs-demo.wav
```

输出验收目标为 24 kHz 单声道 PCM16 WAV。**原来的 `max_frames` 改为 `max_new_tokens`**，调用方必须更新；旧的离线 `generate.py` 已移除，使用上游 HTTP API，避免另外加载一份模型。

请求字段及错误语义遵循上游 Omni；不再承诺旧适配器的 1000 字符/2048 上下文限制或达到帧上限必定拒绝截断。此 profile 的 stage 0 上下文为 8192，默认 token 上限为 2048；长文本要明确设置合理上限并检查生成是否完整。

当前仅验证普通文本、非流式 WAV；未验证声音克隆、其他编码、HTTP 流式、取消或生产并发，不能据此承诺这些功能。

## 逐模型 DynamicEngine 敏感度选择

默认 `HIGGS_METAL_QUANTIZATION=off`；新参数只接受 `off|dynamic`。未设置模型参数时 fallback 到 `VLLM_OMNI_METAL_QUANTIZATION`，模型显式 off 可覆盖全局 dynamic。profile 路径优先 `HIGGS_METAL_QUANTIZATION_PROFILE`，否则 `VLLM_OMNI_METAL_QUANTIZATION_PROFILE`。dynamic 必须显式提供严格批准 profile，缺失/无效/过期一律拒绝，没有 auto、on 或旧 W8-only 生产入口。

候选仅为 **Code2Wav 的 `fc2`**，位于 RVQ decode 后、acoustic decoder 前；Talker、采样、codec chunk/context、拓扑、HTTP 和其它投影/卷积不改。共享 `DynamicEngine.search` 对四个真实 CPU 激活逐一执行源精度、W4、W8，relative L2 与 scaled max 均须满足固定1%，在通过项中选实际存储最小者，否则原精度保留。激活序号是实际 fc2 调用序号，可以跨多个完整 TTS 请求，不要求四次 denoising。捕获上限128MiB，不合成激活，不把局部门槛当作完整TTS批准。

保留实际运行 dtype：FP16/BF16 可选 W4/W8 或源精度 identity；当前 bundled codec 的 fc2 以 FP32 构造时只保留 FP32，**不 cast、不冒充量化**。允许全部原精度保留，不承诺量化收益、听感或加速。`CodecProjection` 保留原 Tensor 返回及 dtype/device 读取合同，并将上游 RVQ transpose 的输入转为连续布局。

`prepare_runtime.py` 从精确 Omni commit 构建正常非 editable 私有 wheel，分发适配器及 `higgs_quantization_preparation.py` 身份资源（离线 CLI 仍执行部署源码）；从相邻后端干净源码构建正常 wheel。准备器/适配器/离线CLI/后端源码均绑定 ready 指纹，安装校验实际位置和 RECORD 摘要/大小。依赖通过显式 `omni-metal-base.pth` 复用基础环境，不宣称全部重新安装。只修改临时 checkout，不修改已安装环境、current、其它服务或历史策略。
安装/ready校验使用选定Python `-I`，包含通用 `quantization.capture`、
`adapter`、`runtime`。共享API可用不代表真实Higgs整任务质量或速度通过，也不会自动启用部署。

请求内不搜索、构造、编译或写 profile。共享 `quantization.runtime.load_profile(path, scope=实际完整执行scope)` 严格检查当前用户 owned0600、4MiB 有界、非符号链接和重复 JSON key，支持普通 schema2 或 `{"profiles":{实际scope:profile}}` 多阶段集合；路径必须为绝对路径。profile 绑定完整本地权重/配置/tokenizer、generation config 内容及不存在状态、适配器/校准器/原生执行源码、部署配置、dtype、环境与候选源权重。旧批准及 engine 源码变化必须重新真实校准，不自动迁移。

### 串行真实校准

先确认没有同模型服务和其它占用资源的工作；本脚本**不会停止服务、下载权重或准备/升级共享环境**。启动需 available ≥16 GiB，运行 available ≥4 GiB、进程树 RSS ≤14 GiB、磁盘保留 ≥2 GiB；单次生成180秒、总阶段3600秒。触线只终止本脚本新建的进程组，不降低门槛。不要另开 HTTP 服务与校准器同时加载两份模型。

从相邻 `vllm-omni-metal` 源码目录执行，使用已经核对的基础环境：

```bash
BASE="$HOME/.local/share/local-model-backend/runtime/c0d736f05e161e0f/bin/python"
RUNTIME="$("$BASE" -I ../docker-compose/higgs-tts-3/prepare_runtime.py)"
"$RUNTIME/bin/python" -I ../docker-compose/higgs-tts-3/prepare_quantization.py \
  --model "$HOME/.cache/modelscope/hub/models/bosonai/higgs-audio-v3-tts-4b" \
  --output "$PWD/outputs/higgs-dynamic-20261022"
```

输出目录必须为空或不存在，重复运行换一个新目录，不能覆盖旧证据。需要完整本地原精度 Higgs checkpoint（当前本地 `model.safetensors` 约8.7 GB，随包 codec）；`prepare.py` 仍拒绝预量化 checkpoint，不能靠放松检查伪造开关。其它已验证基础环境可显式替换 `BASE`，不自动切换 `current`。

准备器固定中英文文本、seed42、`max_new_tokens=512` 和原生采样；先运行四个完整原精度请求采集真实激活，再空闲 RPC 敏感度搜索。每路独立 warmup，然后 float→dynamic→dynamic→float 交错完整生成。要求真实24kHz单声道、有限非静音合法幅度、自然 stop（截断不批准）、pending清零、同路径同seed原始浮点 dtype/shape/bytes 可复现、候选（包括identity）实际 calls/rows >0，以及**完整最终浮点波形 relative L2 与峰值归一 max error 均 ≤1%**。原始 NPY/WAV、完整输出哈希、时延/调用数、comparison 和 supervisor 资源记录保存在新证据目录；不将参考音频或伪造元数据当作真实输出。

这两个文本/seed的质量结论不外推到任意内容、声音克隆、长文本或不同采样设置；后者仍需另外真实验收。`fc2` 是 shape-polymorphic 且无编译的算路，512仅是本次校准样本预算，不是已证明的算子支持上限；**保留原生 HTTP 默认参数及 token 上限，不新增512或显式参数限制**。生产仍执行真实连续布局/dtype/group64算子边界检查。

通过初步门槛后释放原始 float 模块，并用 module/weight 弱引用核验实际释放；identity 候选单独拥有同dtype副本用于这项离线验收。再次完整生成两条文本并与候选逐位一致，最后仅重读选中源权重，以原 engine 搜索精度及审计重建批准摘要。引擎关闭且 supervisor 成功退出后才原子 no-replace 发布0600的 `approved-profile.json`。失败不发布。不宣称听评/ASR、并发或性能验收；PCM16相同不等于原始浮点无损。

本次源码迁移未运行真实模型/重建环境/启动服务，没有新的质量通过或量化收益结论。仅当新的完整证据目录确实发布批准产物后，显式启用（保持相同私有环境、权重和部署配置），再执行真实 HTTP 验收：

```bash
mkdir -p "$HOME/Library/Application Support/higgs-tts-3"
cp -n outputs/higgs-dynamic-20261022/approved-profile.json \
  "$HOME/Library/Application Support/higgs-tts-3/approved-profile.json"
cd ../docker-compose/higgs-tts-3
HIGGS_METAL_QUANTIZATION=dynamic \
HIGGS_METAL_QUANTIZATION_PROFILE="$HOME/Library/Application Support/higgs-tts-3/approved-profile.json" \
VLLM_BACKEND_PYTHON="$BASE" ./start.sh
```

生产加载日志记录实际 `fc2` 位宽/identity 和 profile 摘要；首次 forward 另记录实际 calls/rows，加载成功不等于真实命中。`higgs_quant_snapshot` 返回调用计数（需同一私有 RPC extension）。全局 dynamic 若同时用于 Talker 和 codec，应使用按各自完整scope索引的 profile 集合；codec 专用 profile 不批准其它阶段。公开接口仍为 `/v1/audio/speech`，不新增量化 HTTP 开关。


## 显式 ANE + DynamicEngine 联合执行（默认 off）

`HIGGS_METAL_JOINT=off|joint` 是独立的私有联合模式，默认 off，不继承旧 codec 批准。
它通过私有 Omni wheel 的 **Talker 完成 `load_weights` 后** hook 接入，不覆盖通用 registry。
当前真实 Talker 是36层 Qwen3，固定选择均匀四层 **0、11、23、35**：
未量化 BF16 packed `gate_up_proj`（19456×2560）使用真正的 ANE 线性图，
`down_proj`（2560×9728）使用四个真实调用 step 0/1/2/3 的 DynamicEngine W4/W8/identity 敏感度搜索。
不拆 packed gate/up，不改变 SiLU、TP1 tuple/skip-bias 合同，其余32层保持原样。
至少一个 down 必须实际采用 W4/W8；全部 identity 不会被称为联合量化。
**Code2Wav 的原生 `fc2` 保持原模块、权重对象、dtype 和布局**；按上游实际加载精度记录
FP16/BF16/FP32，不将 checkpoint 的存储精度误当作运行精度，不 cast 成假量化。
联合模式要求 `HIGGS_METAL_QUANTIZATION=off`，不能拼接旧 dynamic/ANE profile。

先完成上述私有环境准备，并在同一私有 Python 显式准备 ANE 工具；这些命令不加载/停止服务，
不下载权重、不切共享 current。需要本机 Apple Silicon NE、完整本地原精度 checkpoint、
现有 clang++/Swift 和系统 CoreML；无需额外 Python CoreML wheel。所有模型校准必须串行：

```bash
BASE="/绝对路径/已核对基础环境/bin/python"
RUNTIME="$("$BASE" -I ../docker-compose/higgs-tts-3/prepare_runtime.py)"
"$RUNTIME/bin/python" -I -m vllm_omni_metal.ane.build --bridge
"$RUNTIME/bin/python" -I ../docker-compose/higgs-tts-3/prepare_joint.py \
  --model "$HOME/.cache/modelscope/hub/models/bosonai/higgs-audio-v3-tts-4b" \
  --output "$PWD/outputs/higgs-joint-new-evidence"
# 等价底层入口：prepare_quantization.py --joint（其它参数相同）
```

资源门槛与上文相同：启动16GiB、运行4GiB headroom、进程树RSS14GiB、磁盘2GiB，
请求180秒/总阶段3600秒。离线实际准备四张 NE-preferred 图，保留源 BF16，
只对 ANE 计算显式授权 FP16 近似；每张图的完整真实投影输出仍须同时满足双1%误差。
ANE 每次仅分担一个256行 chunk 且至少保留一行 GPU，不 padding、不重复 token。
**固定双语短输入可能所有调用 rows≤256，此时必须拒绝零 ANE 命中并保留失败证据**；
不能通过打开 flags、加长/填充校准样本或复用历史批准“通过”本策略。

校准保留原生 Talker→Code2Wav 双语完整生成、自然 stop、24kHz 单声道/幅度/非静音门禁；
每路 warmup 后 float→joint→joint→float，完整浮点波形双1%、同seed逐位重复，
每条文本实际 ANE chunks>0 且真实量化 down calls>0。另以相同 dynamic down、
原生 GPU gate-up 作完整双语 quant-only→joint→joint→quant-only 交错计时和波形双1%检查；
至少两组计时、至少3%收益，且所有 joint 时延优于所有 quant-only 时延。
不把量化本身的收益记到 ANE，不以投影微基准代替整模型计时。
搜索不写全局 ANE 策略：先构造严格暂存 records，再以 `_benchmark=False` 不可变
staged policy 重放；释放 float down baseline，以 module/weight 弱引用核验后再次完整生成，
要求与候选逐位一致。随后显式关闭 ANE handles，关闭 Omni engine；
只有独立 supervisor 成功且资源记录完成，才发布 ANE policies 和 owned0600、原子
no-replace 的最终 `approved-profile.json`。缺少最终联合批准时生产始终拒绝激活。
完整 scope 绑定全 checkpoint、配置/tokenizer/generation config、部署策略、校准器、
Talker/量化/ANE 实现和私有 wheel；最终联合 profile 同时绑定图 identity 与全部实际 shape records。
任何源码/权重/配置改变都必须重新校准，不拼接独立旧批准。

仅在新联合批准生成后才可显式启动；本次实测未获批，不能使用候选 profile：

```bash
cd ../docker-compose/higgs-tts-3
HIGGS_METAL_JOINT=joint \
HIGGS_METAL_JOINT_PROFILE="/绝对路径/outputs/higgs-joint-new-evidence/approved-profile.json" \
HIGGS_METAL_QUANTIZATION=off VLLM_BACKEND_PYTHON="$BASE" ./start.sh
```

生产不编译/搜索/写策略。已批准 shape 的 split 不可修改，未知大于256行的 prefill 明确拒绝；
单步小行数 decode 仍走 GPU，不伪报 ANE 命中。批准只覆盖这次固定完整输入/seed/配置，
不外推任意新文本、长上下文、听感/ASR、声音克隆、流式或并发。
`higgs_joint.py` 和 `higgs_joint_preparation.py` 均由正常私有 wheel 分发，
源码 SHA、RECORD hash/size、fingerprint 和 ready 与已有资源一同核验。

### 2026-10-22 本次真实联合复验

已用既有0.32依赖和官方 Omni `9534efe802ef79ce52af31847fb144f63d259e90`
正常构建、非 editable 安装私有 wheel，校验 source SHA、RECORD、许可证及依赖；
共享 current、现有服务均未修改。首次真实启动纠正了将 codec 运行精度限死FP32的错误：
实际原生 `fc2` 为BF16，现按模块/权重弱引用及dtype核验原对象不变，每次完整生成后复查。

真实双语 Talker→Code2Wav 生成及四层敏感度搜索已执行，0/11/23/35层 down
均局部选W8。第一条 joint 候选完整自然结束，但656次真实量化调用、ANE chunks=0；
短输入达不到256行分块，按门禁立即拒绝，不填充、不放宽、不发布联合profile。
候选波形148800点，原精度153600点；不能据此宣称整段1%质量通过或速度收益。
证据在相邻后端 `outputs/joint-ane-dynamic-20261022-222638/higgs-real-native-identity/`。
联合默认仍off，没有部署重启、听感/ASR或生产性能验收。

## 文件与日志

- `start.sh` / `stop.sh`：仅管理本机模型 launchctl；plist 内联在 `start.sh` 并原子生成到用户 Application Support，不维护独立 plist 模板。
- `prepare.py`：缓存发现、选择、下载及校验，不加载模型、不生成 plist。
- `metal-deploy.yaml`：原生双阶段 Metal 资源配置。
- `prepare_runtime.py`：独立 owned Omni wheel、内容指纹、ready 和依赖一致性检查。
- `higgs_metal_quantization.py`：Code2Wav load-time 适配、严格 profile 消费与离线安全 RPC。
- `prepare_quantization.py`：完整真实双阶段 A/B、最终波形门禁与独立资源 supervisor；不管理已有服务。
- 代理配置与启停：见根目录 `README.md`。
- 模型日志：`~/Library/Logs/higgs-tts-3-metal.log`
- 错误日志：`~/Library/Logs/higgs-tts-3-metal.error.log`
- 代理日志：在根目录执行 `docker compose logs --tail 50 gateway`

旧0.30/0.31结果仅为历史。当前0.32真实双阶段复验及拒绝原因见上节；本部署变更不自动重启或升级运行中的服务，没有听感/ASR或生产性能批准。

插件 Apache-2.0 不改变 Higgs 权重许可；使用前须核对 Boson 研究/非商业及创作者条款。
