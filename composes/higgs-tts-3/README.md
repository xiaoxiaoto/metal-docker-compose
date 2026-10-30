# Higgs TTS 3 · vLLM-Omni + Omni Metal

Apple Silicon 本机使用 **vllm-omni + vllm-omni-metal**，执行上游原生 **Talker → Code2Wav** 双阶段。本目录不再运行 Docker 代理；统一入口由根目录总代理提供，不在 Linux 容器中加载 Metal 模型。

不再使用旧 `higgs_metal`、mlx-audio 融合适配器或自建 HTTP server；接口、模型采样和 codec 由上游 Omni 提供。
根目录将本模型配置为 **dynamic，空闲300秒卸载**。每次根目录启动（包括默认/gateway/lifecycle）先安全停止所有依赖模型、删除历史共享环境和私有 overlay，再重建共享后端；停止失败不删除，构建失败不回退。`./start.sh higgs` 每次调用本目录 `./start.sh --prepare-only` 重建私有环境和 plist，首次推理才离线加载。权重、下载缓存、量化资产和日志保留；精度/量化参数以本次设置为准。详见[根目录重建合同](../../README.md#每次根目录启动前重建共享环境)。

冷模型不会出现在聚合列表，单模型身份查询返回503且不唤醒；先用根目录 `python3 -I runtime_mgmt/lifecycle_control.py status` 检查登记，再发送下文真实推理curl，完成后复查身份。冷请求在原600秒推理预算之外另留最多600秒加载时间。动态保活只覆盖经总代理的请求，勿旁路直连8095推理。


## 环境

- Apple Silicon macOS 15+、MPS、Homebrew、Python 3.11+（根控制脚本与启动引导）、ModelScope CLI；共享运行时由 uv 托管，要求 Python `>=3.12,<3.13`。根目录总代理另需 Docker Compose。
- `start.sh` 使用 `common_env/prepare_backend.py` 准备的统一隔离环境，再构建 Higgs 独占的正常 Omni wheel overlay。显式设置已验证的 `VLLM_BACKEND_PYTHON` 时子脚本不重复调用共享准备器；否则准备器同样先安全停止依赖模型、删除历史环境并重建。不使用独立 Homebrew HEAD 安装/升级流程。
- 共享依赖仅由根目录 [`pyproject.toml`](../../pyproject.toml) / [`uv.lock`](../../uv.lock) 声明，通过 `uv sync --locked` 非 editable 安装：vLLM/Metal `0.32.0`，Omni 固定 `9842ebc17cf1d9c6b440ecb6a79039162f68a1dd`，Omni Metal 固定 GitHub 仓库 `https://github.com/xiaoxiaoto/vllm-omni-metal.git` 的 `5774df7d720b3a8c6d32035729b42c31e480f346`。共享准备不依赖相邻源码；私有 overlay 的现有源码流程不变。详见[根目录合同](../../README.md)。
- vLLM 与 vllm-metal 也从官方 GitHub 源码安装，分别固定 `vllm-project/vllm` 的 `5df63a7203cdb9328d824a27b682e77c7b9928be` 与 `vllm-project/vllm-metal` 的 `6d6f2c37f8242f1ff1a336534adf6c2d2fe2936d`（v0.32.0）；由 uv 隔离构建非 editable wheel，不使用官方预编译 wheel。vLLM 的 CPU 编译形态只是 Metal 插件底座，不代表模型 CPU fallback。
- 私有准备器要求基础环境中 Omni 的 `direct_url.json` 指向官方 Git 仓库且包含完整 commit；来源或元数据不匹配时拒绝准备，不绕过检查。构建还依赖相邻 `vllm-omni-metal` 源码。
- 新私有环境不运行 ensurepip，通过 `omni-metal-base.pth` 使用共享基础环境的 pip；仍执行原样 `pip check` 和完整 ready 校验。
- `prepare.py` 自动扫描 `~/.cache/modelscope/hub/models/作者/模型/`，只提供完整且未量化的 `higgs_multimodal_qwen3` 缓存。唯一候选自动选择，多个候选编号选择，没有候选时输入 ModelScope 仓库名（如 `bosonai/higgs-audio-v3-tts-4b`），下载 `master` 后重新校验；完整缓存不重新下载。
- 部署名称取所选模型卡仓库名 `/` 后的完整文本并转为全小写，例如 `bosonai/higgs-audio-v3-tts-4b` → `higgs-audio-v3-tts-4b`；`prepare.py` 依次输出部署名称、目录，供启动和健康检查复用。显式目录有 `.mdl` 时从中读取仓库名，否则沿用 `higgs-audio-v3-tts-4b`。更换模型时须同步根目录路由及身份检查。
- `HIGGS_MODEL_DIR` 显式指定本地目录时优先使用并校验，不扫描、不下载；无交互 stdin 且需要选择时明确报错。`.mdl` 只解析字符串，不反序列化执行缓存代码。模型路径、索引及所有 safetensors 分片必须完整，量化模型及其他架构不会进入候选列表。
- 软件准备保留官方联网预检，但不会随 Omni main 自动升级；MPS 与 `pip check` 等安装验证通过后才激活共享环境。模型只在准备阶段下载，launchctl 离线加载。构建失败不恢复旧环境，模型缓存保留。
- 根目录 `./start.sh higgs` / `all` 每次传入本轮重建的 `VLLM_BACKEND_PYTHON`，重新生成所选模型的私有环境和 plist，不再复用旧动态部署。

`metal-deploy.yaml` 保留上游 Higgs profile 的原生 sampling、connector 和 async chunk。两个阶段设置 `max_num_seqs=1`、`CUSTOM` MPS SDPA、同步调度和关闭 prefix caching；Code2Wav 模型专用动态量化适配只存在于 Higgs 私有 Omni wheel。每阶段 `gpu_memory_utilization=0.3`，KV 分配保留系统内存 headroom。
Metal SHM 适配遵守上游前缀资源回收合同；逐 key 清理返回是否实际 unlink，统计不修改 stage、音频或 chunk 语义。

## 启停与迁移

只准备、不立即加载：

```bash
cd composes/higgs-tts-3
./start.sh --prepare-only
```

准备模式保留磁盘检查；16GiB可用内存门槛只约束实际模型加载，HTTP冷启动仍执行生命周期资源门禁。

`--prepare-only` 执行依赖检查、模型发现/下载校验、环境准备及 plist 生成，通过 `plutil -lint` 后退出，不执行 `launchctl bootstrap`。本模型的注册任务或 8095 冲突仍拒绝；未提供 `VLLM_BACKEND_PYTHON` 时会调用共享准备器，安全停用全部依赖模型并删除历史环境后重建。未知、重复或多余参数非零退出。无参数仍准备并立即加载：

```bash
cd composes/higgs-tts-3
# 已注册旧任务或需要重新加载配置时，必须先停止：
./stop.sh
./start.sh
curl -fsS http://127.0.0.1:8095/health
# 根目录 ./start.sh 启动总代理后：
curl -fsS 'http://127.0.0.1/v1/models?model=higgs-audio-v3-tts-4b'
```

本机模型监听 `127.0.0.1:8095`，统一入口为根目录 `http://127.0.0.1/v1/`，按请求的 `model` 路由。`start.sh` 拒绝复用已注册的旧任务或杀掉占用端口的其他进程；下次启动生成新的 launchctl plist，直接执行选定安装环境的 `bin/vllm-omni serve --omni`，不依赖源码目录或 shell 的 venv 激活状态。

launchctl label 为 `com.local.higgs-tts-3-metal`，配置在 `~/Library/Application Support/higgs-tts-3/`，不是登录自启项。`stop.sh` 只停止本模型。根目录启动或直接调用共享准备器会安全停止全部依赖模型并重建环境；不修改 Homebrew 或其他项目。

Omni `/health` 成功时没有旧服务的 JSON 字段；模型身份通过 `/v1/models` 检查。health 就绪不等于语音生成验收通过。

仅供本机使用，无鉴权，不要暴露公网。Nginx保留16KiB请求体限制和关闭代理缓冲；生命周期代理按配置限制请求期限，网关额外容纳冷启动等待。

## 生成语音

```bash
curl --fail-with-body --show-error --connect-timeout 5 --max-time 10 \
  'http://127.0.0.1/v1/models'

curl --fail-with-body --show-error --connect-timeout 5 --max-time 10 \
  --get 'http://127.0.0.1/v1/models' \
  --data-urlencode 'model=higgs-audio-v3-tts-4b'

curl --fail-with-body --show-error --connect-timeout 5 --max-time 600 \
  'http://127.0.0.1/v1/audio/speech' \
  --header 'Content-Type: application/json' \
  --data-binary '{"model":"higgs-audio-v3-tts-4b","input":"你好，欢迎使用语音合成。","response_format":"wav","stream":false,"seed":42,"max_new_tokens":512}' \
  --dump-header /tmp/higgs-test.headers --output /tmp/higgs-test.wav &&
afplay /tmp/higgs-test.wav
```

输出验收目标为 24 kHz 单声道 PCM16 WAV。**原来的 `max_frames` 改为 `max_new_tokens`**，调用方必须更新；旧的离线 `generate.py` 已移除，使用上游 HTTP API，避免另外加载一份模型。

请求字段及错误语义遵循上游 Omni；不再承诺旧适配器的 1000 字符/2048 上下文限制或达到帧上限必定拒绝截断。此 profile 的 stage 0 上下文为 8192，默认 token 上限为 2048；长文本要明确设置合理上限并检查生成是否完整。

当前仅验证普通文本、非流式 WAV；未验证声音克隆、其他编码、HTTP 流式、取消或生产并发，不能据此承诺这些功能。

## 逐模型 DynamicEngine 敏感度选择

默认 `METAL_MODEL_OPTIMIZERS` 为空且 `METAL_QUANTIZATION=off`。启用 dynamic 必须显式选择 `dynamic_quantization` 并提供绝对路径 `METAL_QUANTIZATION_PROFILE`；缺失、无效或过期批准一律拒绝。Higgs Omni 当前拒绝 `ane`、`streaming`、`gpu_ops`，不把基础 Metal 的 provider 当成 Omni 支持。内部固定旧后端 ANE 为 off，用户不能通过旧变量启用。

候选仅为 **Code2Wav 的 `fc2`**，位于 RVQ decode 后、acoustic decoder 前；Talker、采样、codec chunk/context、拓扑、HTTP 和其它投影/卷积不改。共享 `DynamicEngine.search` 对四个真实 CPU 激活逐一执行源精度、W4、W8，relative L2 与 scaled max 均须满足固定1%，在通过项中选实际存储最小者，否则原精度保留。激活序号是实际 fc2 调用序号，可以跨多个完整 TTS 请求，不要求四次 denoising。捕获上限128MiB，不合成激活，不把局部门槛当作完整TTS批准。

保留实际运行 dtype：FP16/BF16 可选 W4/W8 或源精度 identity；当前 bundled codec 的 fc2 以 FP32 构造时只保留 FP32，**不 cast、不冒充量化**。允许全部原精度保留，不承诺量化收益、听感或加速。`CodecProjection` 保留原 Tensor 返回及 dtype/device 读取合同，并将上游 RVQ transpose 的输入转为连续布局。

`prepare_runtime.py` 从精确 Omni commit 构建正常非 editable 私有 wheel，分发适配器及离线校准器资源（离线 CLI 从 `extensions/vllm_omni/higgs_tts_3/` 加载源码）；从相邻后端干净源码构建正常 wheel。准备器/适配器/离线CLI/后端源码均绑定 ready 指纹，安装校验实际位置和 RECORD 摘要/大小。依赖通过显式 `omni-metal-base.pth` 复用基础环境，不宣称全部重新安装。只修改临时 checkout，不修改已安装环境、current、其它服务或历史策略。
安装/ready校验使用选定Python `-I`，包含通用 `quantization.capture`、
`adapter`、`runtime`。共享API可用不代表真实Higgs整任务质量或速度通过，也不会自动启用部署。

请求内不搜索、构造、编译或写 profile。共享 `quantization.runtime.load_profile(path, scope=实际完整执行scope)` 严格检查当前用户 owned0600、4MiB 有界、非符号链接和重复 JSON key，支持普通 schema2 或 `{"profiles":{实际scope:profile}}` 多阶段集合；路径必须为绝对路径。profile 绑定完整本地权重/配置/tokenizer、generation config 内容及不存在状态、适配器/校准器/原生执行源码、部署配置、dtype、环境与候选源权重。旧批准及 engine 源码变化必须重新真实校准，不自动迁移。

### 串行真实校准

先确认没有同模型服务和其它占用资源的工作；本脚本**不会停止服务、下载权重或准备/升级共享环境**。启动需 available ≥16 GiB，运行 available ≥4 GiB、进程树 RSS ≤14 GiB、磁盘保留 ≥2 GiB；单次生成180秒、总阶段3600秒。触线只终止本脚本新建的进程组，不降低门槛。不要另开 HTTP 服务与校准器同时加载两份模型。

校准 CLI 固定读取本部署目录的 `metal-deploy.yaml` 和 `prepare.py`；`--deploy-config` 可显式选择其他配置。

从相邻 `vllm-omni-metal` 源码目录执行，使用已经核对的基础环境：

```bash
BASE="/绝对路径/已核对基础环境/bin/python"
RUNTIME="$("$BASE" -I ../metal-docker-compose/composes/higgs-tts-3/prepare_runtime.py)"
"$RUNTIME/bin/python" -I ../metal-docker-compose/extensions/vllm_omni/higgs_tts_3/prepare_quantization.py \
  --model "$HOME/.cache/modelscope/hub/models/bosonai/higgs-audio-v3-tts-4b" \
  --output "$PWD/outputs/higgs-dynamic-evidence"
```

输出目录必须为空或不存在，重复运行换一个新目录，不能覆盖旧证据。需要完整本地原精度 Higgs checkpoint 和随包 codec；`prepare.py` 拒绝预量化 checkpoint。基础环境必须符合准备器的来源检查，不自动切换 `current`。

准备器固定中英文文本、seed42、`max_new_tokens=512` 和原生采样；先运行四个完整原精度请求采集真实激活，再空闲 RPC 敏感度搜索。每路独立 warmup，然后 float→dynamic→dynamic→float 交错完整生成。要求真实24kHz单声道、有限非静音合法幅度、自然 stop（截断不批准）、pending清零、同路径同seed原始浮点 dtype/shape/bytes 可复现、候选（包括identity）实际 calls/rows >0，以及**完整最终浮点波形 relative L2 与峰值归一 max error 均 ≤1%**。原始 NPY/WAV、完整输出哈希、时延/调用数、comparison 和 supervisor 资源记录保存在新证据目录；不将参考音频或伪造元数据当作真实输出。

这两个文本/seed的质量结论不外推到任意内容、声音克隆、长文本或不同采样设置；后者仍需另外真实验收。`fc2` 是 shape-polymorphic 且无编译的算路，512是校准样本预算，不是已证明的算子支持上限；**保留原生 HTTP 默认参数及 token 上限，不新增512或显式参数限制**。生产仍执行真实连续布局/dtype/group64算子边界检查。

通过初步门槛后释放原始 float 模块，并用 module/weight 弱引用核验实际释放；identity 候选单独拥有同dtype副本用于这项离线验收。再次完整生成两条文本并与候选逐位一致，最后仅重读选中源权重，以原 engine 搜索精度及审计重建批准摘要。引擎关闭且 supervisor 成功退出后才原子 no-replace 发布0600的 `approved-profile.json`。失败不发布。不宣称听评/ASR、并发或性能验收；PCM16相同不等于原始浮点无损。

仅当完整证据目录确实发布批准产物后，显式启用（保持相同私有环境、权重和部署配置），再执行真实 HTTP 验收：

```bash
mkdir -p "$HOME/Library/Application Support/higgs-tts-3"
cp -n outputs/higgs-dynamic-evidence/approved-profile.json \
  "$HOME/Library/Application Support/higgs-tts-3/approved-profile.json"
cd ../metal-docker-compose/composes/higgs-tts-3
METAL_MODEL_OPTIMIZERS=dynamic_quantization METAL_QUANTIZATION=dynamic \
METAL_QUANTIZATION_PROFILE="$HOME/Library/Application Support/higgs-tts-3/approved-profile.json" \
VLLM_BACKEND_PYTHON="$BASE" ./start.sh
```

生产加载日志记录实际 `fc2` 位宽/identity 和 profile 摘要；首次 forward 另记录实际 calls/rows，加载成功不等于真实命中。`higgs_quant_snapshot` 返回调用计数（需同一私有 RPC extension）。全局 dynamic 若同时用于 Talker 和 codec，应使用按各自完整scope索引的 profile 集合；codec 专用 profile 不批准其它阶段。公开接口仍为 `/v1/audio/speech`，不新增量化 HTTP 开关。


## ANE + DynamicEngine 联合离线诊断（不可部署）

旧模型私有联合生产入口已移除，现有联合 profile 不符合公共 ANE profile 合同，生产选择 `ane` 会明确拒绝。以下仅描述隔离离线诊断，不代表生产支持。
它通过私有 Omni wheel 的 **Talker 完成 `load_weights` 后** hook 接入，不覆盖通用 registry。
当前真实 Talker 是36层 Qwen3，固定选择均匀四层 **0、11、23、35**：
未量化 BF16 packed `gate_up_proj`（19456×2560）使用真正的 ANE 线性图，
`down_proj`（2560×9728）使用四个真实调用 step 0/1/2/3 的 DynamicEngine W4/W8/identity 敏感度搜索。
不拆 packed gate/up，不改变 SiLU、TP1 tuple/skip-bias 合同，其余32层保持原样。
至少一个 down 必须实际采用 W4/W8；全部 identity 不会被称为联合量化。
**Code2Wav 的原生 `fc2` 保持原模块、权重对象、dtype 和布局**；按上游实际加载精度记录
FP16/BF16/FP32，不将 checkpoint 的存储精度误当作运行精度，不 cast 成假量化。
联合诊断要求 `METAL_QUANTIZATION=off`，不能拼接旧 dynamic/ANE profile。

先完成上述私有环境准备，并在同一私有 Python 显式准备 ANE 工具；这些命令不加载/停止服务，
不下载权重、不切共享 current。需要本机 Apple Silicon NE、完整本地原精度 checkpoint、
现有 clang++/Swift 和系统 CoreML；无需额外 Python CoreML wheel。所有模型校准必须串行：

```bash
BASE="/绝对路径/已核对基础环境/bin/python"
RUNTIME="$("$BASE" -I ../metal-docker-compose/composes/higgs-tts-3/prepare_runtime.py)"
"$RUNTIME/bin/python" -I -m vllm_omni_metal.ane.build --bridge
"$RUNTIME/bin/python" -I ../metal-docker-compose/extensions/vllm_omni/higgs_tts_3/prepare_joint.py \
  --model "$HOME/.cache/modelscope/hub/models/bosonai/higgs-audio-v3-tts-4b" \
  --output "$PWD/outputs/higgs-joint-new-evidence"
# 等价底层入口：../metal-docker-compose/extensions/vllm_omni/higgs_tts_3/prepare_quantization.py --joint（其它参数相同）
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

即使离线诊断生成联合批准，也不能通过当前部署启用；需要匹配公共 ANE 执行和 profile 合同后才能支持生产。

诊断对已批准 shape 固定 split，未知大于256行的 prefill 明确拒绝；
单步小行数 decode 走 GPU。此诊断不外推任意新文本、长上下文、听感/ASR、声音克隆、流式或并发。
`higgs_joint.py` 和 `higgs_joint_preparation.py` 均由正常私有 wheel 分发，
源码 SHA、RECORD hash/size、fingerprint 和 ready 与已有资源一同核验。

## 文件与日志

- `start.sh` / `stop.sh`：仅管理本机模型 launchctl；plist 内联在 `start.sh` 并原子生成到用户 Application Support，不维护独立 plist 模板。
- `prepare.py`：缓存发现、选择、下载及校验，不加载模型、不生成 plist。
- `metal-deploy.yaml`：原生双阶段 Metal 资源配置。
- `prepare_runtime.py`：独立 owned Omni wheel、内容指纹、ready 和依赖一致性检查。
- `extensions/vllm_omni/higgs_tts_3/`：Higgs load-time 适配器、联合适配器及离线校准 CLI/源码。
- 模型日志：`~/Library/Logs/higgs-tts-3-metal.log`
- 错误日志：`~/Library/Logs/higgs-tts-3-metal.error.log`
- 代理日志：在根目录执行 `docker compose logs --tail 50 gateway`

插件 Apache-2.0 不改变 Higgs 权重许可；使用前须核对 Boson 研究/非商业及创作者条款。
