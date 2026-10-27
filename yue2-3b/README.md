# YuE2-3B · vLLM-Omni / Omni Metal 部署

参考 [模型卡](https://huggingface.co/m-a-p/YuE2-3B) 和 `../higgs-tts-3/`。独立端口、服务和缓存，不修改 Higgs / H2O 或通用 Metal 后端。

**源码部署合同（2026-10-22）：量化默认 `off`，仅接受 `off/dynamic`；旧 `manual-w8a16` 启动接口已废除。dynamic 必须使用完整离线搜索批准的私有 profile，且 ANE off。本次源码切换不修改已安装 overlay、ready、plist、用户策略或运行服务；下文旧 W8 测量仅为历史证据，不代表新 dynamic 已通过真实生成验收。**
根目录将本模型配置为 **dynamic，空闲300秒卸载**。每次根目录启动（包括默认/gateway/lifecycle）先安全停止所有依赖模型、删除历史共享环境和私有 overlay，再重建共享后端；停止失败不删除，构建失败不回退。`./start.sh yue2` 每次调用本目录 `./start.sh --prepare-only` 重建私有环境和 plist，首次推理才离线加载。权重、下载缓存、量化资产和日志保留；精度/量化参数以本次设置为准。详见[根目录重建合同](../README.md#每次根目录启动前重建共享环境)。

冷状态的聚合列表不含本模型，单模型身份查询返回503且不唤醒；先在根目录查看 `python3 -I lifecycle_control.py status`，发送下文真实音频请求，完成后再检查身份。首次请求在热推理预算之外另留最多600秒加载时间。动态请求只走统一入口，不直连8096；完整音频/流结束前不会启动空闲计时。

此前自动部署曾因官方 vLLM/Metal 0.32.0 超出旧验证组合0.31.0而中止；该记录不再代表当前环境。现已安装0.32兼容栈，但加载成功仍不能替代真实推理；本次启动排障和验收结果见下方。

### 本次启动故障修复与真实验收（2026-10-22）

原命令可登记dynamic，但首次真实请求因旧
`vllm_metal.pytorch_backend.tensor_bridge` 导入返回HTTP500。通用修复位于
相邻 `vllm-omni-metal`，统一迁移0.32桥接路径，消费分页attention返回值，
并补齐native launcher的reducer/prefill绑定；没有降低精度或关闭报错绕过。

通过既有 `VLLM_OMNI_METAL_WHEEL` 入口，在共享
`runtime/0e63615c03ef50e1` 依赖上新建私有overlay `73cdca6cd88d4c33`；
共享current和其他模型不变。新wheel保存在
`~/.local/share/local-model-backend/wheels/yue2-bridge-032/`，
SHA256 `aad95166b47df8b4785d493853d506f754a206d606d9dcad6f9a903dc086cb0a`。
45个随包文件逐一匹配wheel/RECORD，本次5个变更实现匹配源码，隔离解释器
实际MPS BF16桥接、native launcher、`pip check` 和 `check_metal.py` 均通过。
共享依赖仍通过`.pth`复用，不是全依赖重装。官方联网预检确认核心0.32.0；
Omni最新main已前移，未借本次修复迁移共享环境。

再次执行根目录 `./start.sh yue2` 成功，通过统一入口首次真实请求触发加载；
生成后聚合与单模型身份均含`yue2-3b`。两次同输入/seed42、200帧请求均HTTP200，
耗时39.583秒（含冷启动）与19.883秒，WAV逐字节一致：
48kHz双声道PCM16、383936帧、7.999秒、峰值0.50815、RMS0.07010。
音频SHA256 `da7bed50bc89c12477a3531f76cbcbd3eb0b3c2614559f6b0e8384996a5239b8`。
响应为`audio/wav`，`X-Audio-Truncated:true`，只证明短输出成功，
不证明完整歌曲、歌词准确率或听感质量；未人工听评。

试听：`afplay /tmp/yue2-fixed-test.wav`。对应响应头在
`/tmp/yue2-fixed-test.headers`，复测音频在`/tmp/yue2-fixed-repeat.wav`。
启动目标仍为根目录`yue2`，curl的规范模型ID仍为`yue2-3b`，
按下方“生成歌曲”示例复测；冷模型先发送推理，再核对身份。
模型保持dynamic，空闲300秒后自动卸载。
通用修复与回归提交为相邻仓库`062ce57`，未推送；部署文档和运行资源留在本部署目录。


## 原生拓扑和版本

- YuE2 是单阶段 `AR（semantic tokens）→ terminal NAR flow matching → VAE`，不是 Higgs 的 Talker → Code2Wav 双阶段，不能复用其 connector 或 codec 配置。
- 共享依赖仅由根目录 [`pyproject.toml`](../pyproject.toml) / [`uv.lock`](../uv.lock) 声明，使用 `uv sync --locked` 非 editable 安装：vLLM / vllm-metal `0.32.0`，Omni 固定 `9842ebc17cf1d9c6b440ecb6a79039162f68a1dd`，Omni Metal 来自 `https://github.com/xiaoxiaoto/vllm-omni-metal.git` 的 `5774df7d720b3a8c6d32035729b42c31e480f346`。共享准备不依赖相邻源码；官方联网预检不决定新 commit、不自动升级 main，MPS 与 `pip check` 等验证通过后才激活。每次根目录启动都重建共享环境，私有准备器仍保留自身源码流程与来源兼容门禁。不要混用历史 overlay 或桥接模块布局；详见[根目录合同](../README.md)。
- vLLM 与 vllm-metal 也从官方 GitHub 源码安装，分别固定 `vllm-project/vllm` 的 `5df63a7203cdb9328d824a27b682e77c7b9928be` 与 `vllm-project/vllm-metal` 的 `6d6f2c37f8242f1ff1a336534adf6c2d2fe2936d`（v0.32.0）；由 uv 隔离构建非 editable wheel，不使用官方预编译 wheel。vLLM 的 CPU 编译形态只是 Metal 插件底座，不代表模型 CPU fallback。
- checkpoint 的架构名是 `YuE2ForCausalLM`，profile 用 `hf_overrides.architectures: [Yue2ForCausalLM]` 选择 Omni 原生类，不执行 checkpoint remote code。
- YuE2 的实际文本 tokenizer 是本地 `qwen.tiktoken`，engine 接收 token IDs 并设置 `skip_tokenizer_init: true`。本 profile 显式用 `tokenizer: null` 覆盖上游无用的 `gpt2` 占位引用，避免 vLLM 在离线模式下仍尝试解析 GPT-2 缓存；不额外下载 GPT-2，也不关闭离线限制。
- 同时需要未量化 `m-a-p/YuE2-3B`（AR/NAR BF16，约 7.26 GB）和标准 `m-a-p/YuE2-Vae`（FP32，约 0.53 GB）；输出目标是 **48 kHz 双声道**。默认使用 ModelScope 完整下载并保留仓库文件/缓存元数据，不安装仓库内的推理 wheel；显式选择 HF 时只下载模型必需文件。
- 单请求默认 profile；Metal 使用 CUSTOM MPS SDPA、同步 uni 和关闭 prefix cache；保留 `max_model_len=24576`，KV恢复量化前的 `gpu_memory_utilization=0.30` 自动预算，不再固定3GiB。NAR/VAE 在 KV 预算之外分配，不能把该比例视为总内存上限；实际 bootstrap 前至少16GiB available、部署保留2GiB磁盘，不提高 MPS 水位。`--prepare-only` 不受真实模型加载的16GiB门槛限制。

## Apple Silicon / Metal

需要 Apple Silicon、macOS 15+、Homebrew、Python 3.11+（根控制脚本与启动引导）、ModelScope CLI；uv 托管的共享运行时要求 Python `>=3.12,<3.13`。根目录总代理另需 Docker Compose。与 Higgs 相同：模型在 macOS launchctl 中运行，本目录不再运行 Docker 代理，不能在 Linux 容器中使用 Metal。

`patches/yue2-mps.patch` 对固定 Omni 源码进行小范围修复，保留原生采样、NAR/VAE 算法及错误处理：

1. 请求 generator 使用输入张量的设备，MPS 张量不再配 CPU generator。
2. 只有 CUDA 使用 pinned memory / event；MPS 同步读回，仍保留上游的延迟接受 token、重算与结束语义。
3. VAE 初始化和同步解码跟随模型参数的设备，MPS 解码保持 FP32，最终音频搬回 CPU。

`prepare_runtime.py` 从共享环境记录的官方 commit 构建 **YuE2 私有 Omni wheel**，安装到 `~/.local/share/local-model-backend/overlays/yue2-3b/<fingerprint>/`，版本标记为 `0.32.0.dev0+cpu.yue2mps2`。不修改共享后端已安装的 Omni 文件、通用 Metal 后端或模型 registry，不复制第二套模型实现，不模拟 CUDA、不启用 CPU fallback。

私有环境覆盖 Omni；显式 `VLLM_OMNI_METAL_WHEEL=/absolute/backend.whl` 还会仅在该 overlay 安装通用后端 wheel，Torch / vLLM / vllm-metal 依赖仍通过 `.pth` 复用冻结基础环境，不修改共享后端或 `current`。指纹包含补丁、固定 commit、基础路径、Python版本、准备器、`yue2_dynamic_quantization.py`、`prepare_dynamic_quantization.py`、共享后端源码/native资源及所选 wheel 的路径与内容。生产私有模块通过 Omni wheel 的 `py-modules` 正常分发，不在安装后复制进 site-packages；ready 在选定 Python `-I` 下检查源码 SHA、overlay 安装归属、RECORD、通用 `quantization.engine/linear` 导入和实际 Torch-stream launcher。已有 ready 也重新检查，失败撤销 ready。所选 wheel 路径保存为状态目录中的 `backend-wheel`；已选择但丢失时拒绝启动，不静默退回旧包。旧 overlay 保留，不在引擎启动/请求期间 JIT。

通用后端源码已按功能分包：GPU 算子位于 `vllm_omni_metal.ops`，ANE 运行时为 `vllm_omni_metal.ane.runtime`，离线构建入口为 `vllm_omni_metal.ane.build`；准备器与 ready 校验使用这些新路径，源码／资源指纹递归包含子包。本次仅同步源码入口，不代表当前运行服务或已安装环境已迁移；没有运行准备器、切换 ready／current、重启服务或启用量化／ANE。

### 冻结旧后端到 dynamic 的实际迁移修复（2026-10-22）

旧冻结后端仍是平铺 `ane.py`，准备器曾在安装所选新 wheel 之前导入
`ane.runtime`，导致显式指定新 wheel 也无法迁移。现已删除这项重复的基线 API
预检；基线版本检查保留，新 API、私有源码 SHA、RECORD 和 launcher 仍在安装后
严格校验，不修改共享冻结环境。

本次从冻结基线 `c0d736f05e161e0f`（Omni `b164fea797531c1fa10f8b87829632cad006b9fc`）
实际构建并安装新私有环境 `5ec6aec8d7bd7d3c`，选用独立后端 wheel SHA256
`efe32db88a1b4732849896214ecca1a7d506e68dd0c17e04150ec23b572e7472`。
`pip check`、安装归属／源码／RECORD、新 API 和实际 launcher 加载通过；
`check_metal.py` 的真实 MPS RNG、合批隔离、清理、tiny NAR／FP32 VAE 自检通过，
针对性回归98项通过、1项跳过。以上不代表真实歌曲或动态量化质量通过。

旧 `manual-w8a16` 服务 health503，已仅通过本目录 `stop.sh` 停止；
旧 plist 保留为状态目录下的 `*.manual-w8a16-disabled-*.plist`，移出活动路径，
避免根目录生命周期管理器再次激活旧 W8；策略、模型、媒体和旧 overlay 均保留。
真实 `prepare_dynamic_quantization.py --frames 512 --candidate-layers auto`
在模型启动前因 available10.3584GiB低于16GiB拒绝，尚缺约5.6416GiB。
显式 `YUE2_METAL_QUANTIZATION=dynamic ./start.sh` 也拒绝缺少批准 profile；
没有回退off、伪造批准或启用新服务。其他模型和共享 `current` 未改变。
完整证据位于相邻后端 `outputs/yue2-dynamic-repair-20261022/`；
释放足够内存后，须重新使用空证据目录完成真实校准，再显式执行 dynamic 启动。

只准备、不立即加载：

```bash
cd docker-compose/yue2-3b
./start.sh --prepare-only
```

准备模式执行环境准备、版本检查、`check_metal.py` tiny MPS 自检、模型下载/验证及 plist 生成，通过 `plutil -lint` 后退出，不 bootstrap 或加载真实模型。16GiB available 仅在实际 bootstrap 前检查，2GiB磁盘门槛保留。已有注册任务或 8096 冲突仍拒绝；未提供 `VLLM_BACKEND_PYTHON` 时调用共享准备器，安全停用全部依赖模型并删除历史环境后重建。非法参数非零退出；无参数仍准备后立即加载：

```bash
cd docker-compose/yue2-3b
./start.sh  # 自检通过后准备 ModelScope 模型并加载本机模型；总代理在根目录单独启动
./stop.sh
```

`start.sh` 未收到 `VLLM_BACKEND_PYTHON` 时调用根目录共享准备器，联网预检后安全停止所有依赖模型、删除历史环境并重建，再准备 YuE2 私有 patched Omni overlay。不修改 Homebrew Formula。

根目录 `./start.sh yue2` 每次重新准备私有 overlay 和 plist 后保持冷状态，不再复用旧 plist。准备只发生在显式启动阶段；HTTP 冷启动复用本次已准备部署，不下载、不重建环境。

`check_metal.py` 不再依靠源码字符串判断：实际执行上游 MPS 请求 RNG、完整 sampler / 同步读回、同 seed 与合批隔离、帧上限结束、请求清理、非法分数拒绝，以及随机 tiny NAR 和 FP32 MPS VAE 解码/有限值检查；另检查缺少绑定产物时通用 DenseSwiGLU 保留原生 GPU FFN。它不下载或加载真实大权重，也不是听感验收。通过检查或 `/health` 不等于实际歌曲生成成功。

### 显式离线准备 GPU + ANE

YuE2 私有补丁只在完整 NAR 结构后接入通用 `DenseSwiGLU`，没有模型名路由进入 Metal 后端。`VLLM_OMNI_METAL_ANE=auto` 默认生效，但仅在设备、源码、权重、行数、AR 上下文和执行配置均匹配已批准的全流水线校准时分流；缺少产物仍走 GPU。`off` 可在下次启动时关闭。需要基础插件含 `ane/runtime.py` / `ane/build.py`；普通部署不会编译工具或下载模型。

使用已缓存真实权重执行一次 256–512 帧离线校准（启动前可用 RAM ≥16 GiB，运行时可用 RAM ≥4 GiB、整个子进程树 RSS ≤14 GiB、阶段≤900 秒、请求≤180 秒、磁盘保留≥2 GiB）：

```bash
<selected-backend-python> -m vllm_omni_metal.ane.build --bridge
VLLM_PLUGINS=metal,vllm_omni_register_models <yue2-private-overlay-python> prepare_ane.py --frames 512
```

校准通过新的隔离 worker 完成，不访问或重启 8096/launchd，也不修改默认服务资源。`--deploy-config /path/to/profile.yaml` 可选择经过独立内存测量的资源 profile；普通ANE校准仅注入观测扩展，不改请求、拓扑、attention 或资源值，只有显式 GPU attention A/B 会切换被测路由。使用源码后端时显式传入相应 `PYTHONPATH`，私有 overlay 仍须使用 `yue2mps2` 版本。

共享 RPC 严格要求 `{"supported": true, "result": {...}}`；上游把异常转换成的 `{"supported": false, "error": ...}` 会原样写入 `rpc-error.json` 并终止，**不继续生成、不自动切成 GPU 测量、不假装 ANE 成功**。ANE 准备/构造失败时保留原健康 GPU FFN 组；只有请求已清理且候选 pool/disabled/closed 验证通过后才替换并关闭旧组。批准后的默认组也采用事务替换。

证据保存在系统临时目录 `yue2-ane-*`（macOS 通常位于 `/var/folders/.../T/`）。ANE 校准检查同模式同 seed 音频可复现，并要求整请求候选稳定快至少 3% 才批准默认比例；FP16 ANE 不代表 BF16 等价或无损，需另行听测。准备时间和图磁盘另计，200 帧默认短音频 NAR 只有 202 行，不会协同。当前真实运行曾在 ANE 图自检被拒绝：relative L2 `0.0549945517571671`、max `0.007941411783255007`；该失败没有有效音频，也没有当前安装的加速结论，不能引用旧原型比例作为本次收益。

### 显式 GPU-only 真实测量与 attention A/B

ANE 被拒绝后，需**另起显式命令**测量 GPU，而不是在失败流程内自动继续：

```bash
VLLM_PLUGINS=metal,vllm_omni_register_models <yue2-private-overlay-python> prepare_ane.py \
  --gpu-only --frames 512 --deploy-config /path/to/profile.yaml

# 仅 GPU-only 合法：同 worker 比较旧 MLX bridge 与当前 Torch-stream launcher
VLLM_PLUGINS=metal,vllm_omni_register_models <yue2-private-overlay-python> prepare_ane.py \
  --gpu-only --attention-ab --frames 512 --deploy-config /path/to/profile.yaml
```

GPU-only 固定 512 帧、seed 42、相同歌词/风格/上游固定采样，复用 `AsyncOmni` / `Yue2Adapter` 完整 AR→NAR→VAE 请求。子进程显式设置 `VLLM_OMNI_METAL_ANE=off`，不调用 prepare/set-mode/approve、不发布 ANE 配置，允许健康的非 benchmark GPU 组。普通测量独立 warmup 后完整计时两次；A/B 对两路各 warmup，再按 bridge→Torch→Torch→bridge 正反交错计时。路径切换仅在空闲隔离 worker 内通过显式 benchmark RPC 包装通用 attention，不修改模型方法或生产服务；每请求记录成功的 paged-attention 调用数和实际 Torch launcher forward 次数，缺 launcher 或没有命中不能算优化。

每次生成要求 pending 清零、48 kHz 双声道、有限/非静音/幅度合法且至少五秒音频，并保存 WAV、浮点音频 SHA-256、延迟、seed 和帧预算。`input.json` 保存歌词/风格/prompt/采样，`environment.json` 保存有效 attention 实现、模型/VAE dtype、资源配置与优化产物路径/哈希/存在性；`results.json` 保存 warmup 和计时记录，`supervisor.json` 保存退出/资源停止线观测。只有各路径同 seed hash 一致且两次完整计时均通过，才写 `completed.json`。A/B 单列跨路径 shape、位等价、relative L2/max abs 与中位延迟变化，**跨路径允许数值不同，不能因此声称无损或听感等价**；普通 GPU-only 不产生旧后端相对加速结论。

两种模式保持相同 RAM/RSS/磁盘/超时门槛，不会因为真实模型内存压力、数值门禁或生成超时而缩短请求、降低精度、增大水位或下载替代权重。CPU 回归覆盖小型真实 FFN 的失败后可调用性、事务替换及严格 RPC 序列化，不代表歌曲或性能验收；真实输出须由以上实际权重命令产生并独立听测。

### 显式选择性 W8A16 校准与真实 A/B

`prepare_quantization.py` 复用上述安全 RPC 与资源监督，仅对独立的 NAR
`down_proj` 做 MLX affine W8/group64；AR、采样、packed gate/up、VAE 和原 checkpoint
不变。通过现有 `_metal_ffn` 实例 hook 接入实际执行路径，不改已安装 Omni、registry
或服务配置，不需要重新打模型补丁。需要共享后端提供 `quantization/linear.py`；准备器的
ready 校验检查该 API，源码／资源指纹变化触发正常的新环境准备，但本工具不运行准备器。

从通用后端 checkout 根目录执行，选择已经存在的 `yue2mps2` 私有解释器：

```bash
PYTHONPATH="$PWD" VLLM_PLUGINS=metal,vllm_omni_register_models \
  <yue2-private-overlay-python> ../docker-compose/yue2-3b/prepare_quantization.py \
  --frames 512 --deploy-config /path/to/resource-profile.yaml \
  --output "$PWD/outputs/yue2-w8a16-current"
```

默认均匀选择最多四个 NAR 层的独立 down 投影作为候选，捕获第0/8/16/24次真实调用
的完整激活（总预算128MiB），按最大相对L2／scaled max排名，选择一层；可用
`--candidate-layers 0,9,18,27` 和 `--select-count 1` 显式固定。不是全模型穷尽搜索，
不是每个请求动态排名，也不把权重重建误差等同于最终音频质量。

先真实捕获／审计，再量化独立 warmup，正式顺序为 float→W8→W8→float。两路保持同
worker、歌词、风格、seed42、512帧预算及 attention。分别保存原始NPY、PCM16 WAV和
hash，报告浮点逐位一致、PCM一致、L2／max／SNR、完整延迟、RTF与实际量化calls/rows。
局部1%门禁失败仍可完成明确标记的诊断A/B；只有局部合格、最终浮点逐位一致、同路
可复现且完整生成稳定快超过3%才保存批准profile，不自动激活生产服务。

A/B为切换基线显式保留选中浮点模块；随后释放基线和 `nar_fused` 对应引用，再生成
验证候选输出。模块源／量化字节数与Torch／MLX／RSS观测分别记录，MLX peak为整个
worker生命周期峰值，不是每路请求峰值；该流程不是独立冷启动净RSS／低峰值加载验收。
当前适配在完整模型加载后替换，不声称已经实现流式量化loader。

2026-10-17本次真实运行完成：选中第0层down，局部门槛通过，但最终音频非逐位／PCM
一致，完整生成未证明加速，故未保存批准profile或改变服务默认配置。释放基准浮点后仍
复现量化输出；原始NPY/WAV及逐请求证据在通用后端
`outputs/yue2-w8a16-20261017/`。完整计时、误差与内存口径见
[`docs/generic-backend.md`](../../vllm-omni-metal/docs/generic-backend.md#2026-10-17真实-yue2-选择性-w8a16-对照)。

### 显式固定 W4/W8/BF16 混合精度诊断（不批准）

从通用后端 checkout 根目录运行已有私有解释器，`--precisions` 显式指定最多四个
不同 NAR 独立 down 投影，并且至少各有一个 W4A16、W8A16 和保留 BF16：

```bash
PYTHONPATH="$PWD" VLLM_PLUGINS=metal,vllm_omni_register_models \
  YUE2_METAL_QUANTIZATION=off <private-python> \
  ../docker-compose/yue2-3b/prepare_quantization.py \
  --precisions '{"nar_layers.0.mlp.down_proj":"w4a16","nar_layers.9.mlp.down_proj":"w8a16","nar_layers.18.mlp.down_proj":"bf16"}' \
  --frames 512 --deploy-config <原配置> --output <空证据目录>
```

这是固定显式计划，不捕获激活排名、不做敏感度搜索，也不复用 W8 校准结果来批准
W4。实际替换仍由通用 schema 2 profile API 完成，BF16 项保持原模块／权重身份，
只释放量化项的浮点基线及融合缓存别名。工具诊断原生 AR→NAR→VAE 全链路，但混合
精度只发生在指定 NAR 投影；AR 和其余 NAR 保持原精度、VAE 保持 FP32。
这不是全层混合精度、自动批准、无损或加速声明；profile 和完成记录始终
`approved=false`，局部误差门槛未经搜索评估，`local_pass=null`。

工具先用尚未安装诊断hook的原生FFN保存参考，再验证hook float逐位一致；不一致即停止，
不把仪器改写后的float自证为原精度。随后两路各warmup，正式float→mixed→mixed→float，
并在释放量化项全部浮点引用后再次生成；仅记录实际命中的bits／calls／rows。

同样使用 seed42／512帧、空闲 worker、真实 forward 命中、音频与释放引用证据；
同 worker 释放后的观测不是独立冷启动峰值验收。资源门槛不变：启动 available
至少16GiB、运行 available 至少4GiB、进程树 RSS 至多14GiB、磁盘保留至少2GiB、
请求至多180秒、阶段至多900秒。不足时保留监督阻塞证据，不降门槛、不下载替代
权重、不重启服务或修改 ready／plist／部署默认。上述命令是诊断用法，不表示本次
已取得真实混合精度实测通过；历史 W8 实测与手动部署授权仍仅适用于原范围。

2026-10-22实际启动尝试在模型进程创建前被门禁拒绝：available9.529GiB，要求至少16GiB。
本地真实模型／VAE完整性、私有`yue2mps2`环境、当前schema2源码以及显式构建的
Torch-stream launcher已核对；没有本次真实音频、延迟或误差结果。
证据：通用后端`outputs/yue2-mixed-20261022/{preflight,blocked}.json`。
108项定向回归（含小型MPS hook）和668项非集成回归通过，不代替真实模型对照。
本次没有修改通用插件安装资源或重建／激活overlay，没有改服务、ready、plist或默认量化。

### 生产部署：离线批准的 dynamic profile（默认关闭）

`start.sh` 默认量化off；`YUE2_METAL_QUANTIZATION=off|dynamic` 显式优先于
全局 `VLLM_OMNI_METAL_QUANTIZATION`，未设置时fallback全局，显式off覆盖全局dynamic。
profile优先 `YUE2_METAL_QUANTIZATION_PROFILE`，否则全局
`VLLM_OMNI_METAL_QUANTIZATION_PROFILE`；均须绝对路径。共享
`load_profile(path, *, scope)` 支持单个schema2或 `{"profiles": {实际完整scope: profile}}`
集合；文件必须当前用户owned0600、非符号链接、4MiB内普通文件，dynamic强制ANE off。
完整scope、权重、实现指纹及批准证据仍由生产适配器验证；不能手改approved或
用历史manual policy绕过完整批准，缺失/过期/不支持合同明确报错，不静默fallback。
通用 `vllm_omni_metal.quantization.engine.DynamicEngine` 执行搜索／应用，
模型私有模块只提供 NAR 适配与安装验证；请求期间不搜索、不校准、不写 profile。
生产模块与 `prepare_dynamic_quantization.py` 均作为 Omni wheel 的 `py-modules`
安装在同一目录，后者仅供生产验证准备器源码指纹，不在生产启动时导入；
ready 同时核对其 SHA 与 RECORD 归属。
ready同时导入通用 `quantization.capture`、`adapter`、`runtime` 新API，
所有安装校验使用选定Python `-I`。FP16/BF16保持源dtype，FP32只identity；
通用入口接入不代表YuE2真实动态质量/速度通过，不自动启用或迁移现有环境。

先用所选私有解释器进行完整真实模型离线敏感度搜索；`--output` 必须是空目录：

```bash
PYTHONPATH=/absolute/vllm-omni-metal VLLM_PLUGINS=metal,vllm_omni_register_models \
  YUE2_METAL_QUANTIZATION=off <selected-private-python> \
  prepare_dynamic_quantization.py --frames 512 --candidate-layers auto \
  --output /absolute/empty-dynamic-evidence
```

`candidate-profile.json` 是候选而非部署授权；只有全部完整批准条件通过才输出
`approved-profile.json`。质量／速度／释放浮点后的真实生成等任何门槛失败，
保留拒绝证据，不发布生产 profile。不承诺无损或加速；历史测量不能代替本次批准。

2026-10-22随后实际完成真实校准，证据为相邻后端
`outputs/yue2-dynamic-repair-20261022/calibration-continue-2/`。
新overlay `5ec6aec8d7bd7d3c`，本地真实权重、原资源配置、seed42、512帧，
监督366.80秒成功结束；最低available5.905GiB，进程树RSS最高1.805GiB。
搜索选择NAR第0／9／27层down为W8，第18层保留BF16；所有W4候选拒绝。
原生参考与hook float逐位一致，两路各自同seed可复现；但组合最终音频relative L2
1.3409%、scaled max7.9338%，跨路线float32及PCM16均不一致，也未取得稳定3%加速。
正式float耗时41.7593／41.3499秒，dynamic耗时40.2152／41.9528秒。
三项量化投影每次生成各命中64次／32896行；全部8份实际音频均为48kHz双声道、
20.4787秒、有限非静音。释放旧浮点引用后又真实生成复现，pending清零、engine关闭。
**局部搜索通过，完整批准拒绝**；没有发布approved profile，没有启用服务或回退模式。
512帧结果不是完整歌曲或听感验收，完整数值及边界见通用后端文档的本次实测记录。

在已获准管理该模型、且服务已停止时，显式指定已有基础解释器和新后端 wheel：

```bash
BASE=/absolute/existing/base/bin/python
VLLM_BACKEND_PYTHON="$BASE" VLLM_OMNI_METAL_WHEEL=/absolute/backend.whl \
  YUE2_METAL_QUANTIZATION=dynamic \
  YUE2_METAL_QUANTIZATION_PROFILE=/absolute/empty-dynamic-evidence/approved-profile.json \
  ./start.sh
```

首次成功准备后保存 wheel 选择，下次仍应提供 `VLLM_BACKEND_PYTHON`。
profile 原子复制到 TCC-safe 状态目录，launchd 不依赖 Documents 或 PYTHONPATH。
已注册任务／占用端口仍明确拒绝；随机 tiny 自检强制 quantization=off，
不是量化生产命中。本次仅改源码，不执行上述命令，不触碰现有服务或用户文件。

### 历史废除：手动 W8A16 部署及实测

以下 `manual-w8a16` 启动方式及手动授权协议已废除；旧准备器／适配器不再分发，
原始 JSON、WAV 和测量保留。不可将 `manual_override=true, approved=false`
的历史策略用于 dynamic，也不自动迁移历史授权。

2026-10-19 重新授权后复用已安装 overlay `9c8b39c6da1f48c2` 与原私有策略，不重装上游、
不升级共享后端。当时 launchctl plist 与本地启动器默认均已开启。使用用户原样请求
（两句中文歌词、seed42、cot=off、max_new_tokens=1024）实际返回 HTTP200、
48kHz双声道40.958667秒非静音 WAV，`X-Audio-Truncated:true`；请求新增成功
`YUE2_MANUAL_W8A16_FIRST_FORWARD`，1026行。PCM与本次临时W8对照一致，
盲ASR中的指定16字按序完整，随后有未提供歌词；不声称严格歌词遵循或听感等价。
启动 available21.979GiB，监督最低available7.824GiB、进程树RSS最高1.857GiB，
未触发门禁。证据在相邻后端 `outputs/yue2-w8-enabled-20261019/`；
服务保持量化运行。只更改本地部署默认和plist，未构建新wheel或发布远端。

历史原精度回退命令（仅供记录，本次未执行）：

```bash
./stop.sh
VLLM_BACKEND_PYTHON="$BASE" YUE2_METAL_QUANTIZATION=off \
  VLLM_OMNI_METAL_ANE=off ./start.sh
```

以下为撤销前的历史验收，不代表当前配置：2026-10-17实验 overlay 为 `9c8b39c6da1f48c2`，私有 Omni wheel 正常包含适配器，
通用后端 wheel 仅安装进该 overlay；与既有 YuE2 模型源码相比只有手动入口的7行增量。
非editable环境用 `-I` 核对全部34个后端包文件并实际执行量化；全部非集成回归422项通过。
使用上面明确的 `VLLM_BACKEND_PYTHON` 启动，未切换共享 `current`、未停止其他模型、
未下载权重或重装Homebrew。保留context24576，显式KV3GiB。

8096实际HTTP两次请求分别40.372／39.915秒，均为48kHz双声道PCM16、982976帧，
20.478667秒，同seed WAV和PCM逐字节一致；响应 `X-Audio-Truncated: true` 表示达到512帧
预算，不是完整歌曲自然结束。请求期间新增日志
`YUE2_MANUAL_W8A16_FIRST_FORWARD`：context567、首调用514行、量化calls=1，证明真实请求
走量化而非仅配置存在。HTTP最低available8.363GiB，未触发4GiB／14GiB RSS停止线；
服务验收后仍运行。80端口 `/v1/models` 已确认包含 `yue2-3b`，生成验收直接使用8096。

这两次HTTP不是与原精度的交错A/B，不能据此宣称提速。新采集的离线对照仍有损且未提速，
证据为后端 `outputs/yue2-manual-evidence-20261017/`；部署输入、安装资源校验、
HTTP头／WAV／hash／实际forward日志位于 `outputs/yue2-manual-deployment-20261017/`。
旧部署plist与资源配置已备份在该目录的 `previous-state/`；用户权重和媒体均保留。

用户随后要求还原并报告实际音质很差。本次只关闭量化、不删除代码／策略／权重／媒体，
并恢复量化前资源配置；实际512帧HTTP返回200，48kHz双声道20.478667秒，与离线原精度
float参考的最大差异不超过一个PCM16编码步长，请求期间无量化forward日志。证据在
`outputs/yue2-restored-float-20261017/`，部署边界回归30项通过。不得用先前局部0.3%误差、
波形SNR或同seed重复性反驳用户的听感反馈。

vpipe固定版本 `823f282924bb480d5f2bf9961077afb6cb5534dd` 的YuE2默认量化的是AR投影，
不是NAR：`lm_quant=w8` 加载时量化AR，embedding/head和NAR保留BF16；NAR动态INT8为
默认关闭、明确LOSSY的M5路径。其默认 `cot=full` 会规划旋律／和弦，而此前校准工具使用
`cot=off`、512帧截断。当前Omni HTTP full/melody还要求外部提供ABC，不能只改一个字段
就称为相同流程。通用mixed/AWQ不是此处NARdown选层的依据；局部误差最小不等于听感
不敏感。详见通用后端文档中的撤销与vpipe差异说明。

只检查、暂不下载大权重或更新 Homebrew：

```bash
base="$(brew --prefix xiaoxiaoto/vllm-omni-metal/vllm-omni-metal)/libexec/bin/python"
runtime="$("$base" prepare_runtime.py)"
VLLM_PLUGINS=metal,vllm_omni_register_models "$runtime/bin/python" check_metal.py
```

**Higgs 等大模型运行时，先由其管理者停止，释放统一内存，再启动 YuE2；本脚本不自动停止其他服务。**

脚本准备模型、生成 plist、启动本机 8096 服务。`prepare.py` 依次输出模型卡仓库名 `/` 后的完整文本转为全小写、主模型目录、VAE 目录；`start.sh` 读取部署名称变量，统一用于 `--served-model-name` 和健康检查，VAE 不单独暴露模型 ID。根目录总代理单独启动，通过请求中的 `model=yue2-3b` 转发；统一入口 `http://127.0.0.1/v1/`：

- `YUE2_MODEL_DIR` / `YUE2_VAE_DIR`：可分别显式指定本地目录；无效时直接失败，不下载替代品。
- 默认复用 `~/.cache/modelscope/hub/models/m-a-p/` 中完整缓存；缺失/不完整时通过 `modelscope download --revision master --local_dir ...` 下载到同一目录。不执行 `.mdl` pickle，不删除 `.mdl` / `.msc`、不完整权重或其他缓存文件。
- `YUE2_DOWNLOAD_SOURCE=hf`：显式选择 Hugging Face；仍优先复用完整魔搭缓存，缺失时下载到 `~/.cache/yue2-3b/m-a-p/`。ModelScope 下载失败不会自动启用这个选项。
- `YUE2_CACHE`：显式改变下载根目录；设置后只在该根目录查找默认缓存。
- 两份模型通过绝对本地路径和 `YUE2_VAE` 交给 Omni；模型进程设置 `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`。
- launchctl label：`com.local.yue2-3b-metal`；plist 和运行时复制的 `metal-deploy.yaml` 位于 `~/Library/Application Support/yue2-3b/`，工作目录也在该目录，避免 launchd 访问 Documents 的 macOS 隐私权限限制；不是登录自启项。修改项目中的 profile 后须重新启停，运行时副本会重新复制。
- health 等待允许 launchctl 暂处于 `spawn scheduled` / `never exited` 状态，只在任务已卸载或确实退出后提前失败，不再把待启动任务误判成退出。
- 日志：`~/Library/Logs/yue2-3b-metal.log` / `yue2-3b-metal.error.log`；代理：在根目录执行 `docker compose logs --tail 100 gateway`。
- 已注册任务或 8096 端口被占用时拒绝启动，不杀其他进程。`stop.sh` 只管理本目录的 Metal 任务，不操作总代理。

接口无鉴权，仅供本机；不要暴露公网。根目录总代理保留请求体限制 128 KiB、等待 1800 秒、关闭响应缓冲。

## 生成歌曲

**HTTP API 与 HF quick start 不同：`input` 是歌词，`instructions` 是音乐风格。默认 `cot=off`；`full` / `melody` 必须提供 `extra_params.abc`，speech 接口不自动生成 ABC 计划。** 自动规划可使用上游离线 YuE2 示例，或下述顺序调用既有 completions / speech 接口的独立客户端；不另建 HTTP adapter。

服务加载就绪后，使用真实歌曲请求继续验收：

```bash
curl --fail-with-body --show-error --connect-timeout 5 --max-time 10 \
  --get 'http://127.0.0.1/v1/models' \
  --data-urlencode 'model=yue2-3b'

curl --fail-with-body --show-error --connect-timeout 5 --max-time 1800 \
  'http://127.0.0.1/v1/audio/speech' \
  --header 'Content-Type: application/json' \
  --data-binary '{"model":"yue2-3b","input":"[Verse]\n今晚星光照亮街道\n我们一起追逐微笑","instructions":"Mandarin pop, warm lead vocal, piano and acoustic guitar, at 100 BPM","extra_params":{"cot":"off"},"response_format":"wav","stream":false,"seed":42,"max_new_tokens":200}' \
  --dump-header /tmp/yue2-test.headers --output /tmp/yue2-test.wav &&
afplay /tmp/yue2-test.wav
```

`max_new_tokens` 是 25 fps 的音频帧预算，上游 API 允许 200–9000；200 帧约 8 秒，仅用于短输出检查，可能截断歌词。完整歌曲应提高预算并检查响应头的截断标记。采样遵循上游固定 preset；不支持自选 voice、参考音频或流式。

生成验收应检查 WAV 解码、48 kHz/双声道、有限且非静音、同 seed、截断/终止与取消后的资源回收。听感、歌词正确率、并发和生产性能不能由 health 或短样本推断。

### 原精度完整歌曲基线

当前不再开启量化；AR/NAR 保留 BF16、VAE 保留 FP32、ANE off。先确认原精度完整歌曲，
再对齐真实旋律／和弦规划和自然结束；后续只有重新授权时才研究 AR-only，不量化 NAR/VAE。

`establish_baseline.py` 只连接已经运行的本机服务，不加载第二份权重、不启停服务、不下载、
不自动播放。默认 `--mode all` 串行执行：完整 `cot=off` → YuE2 自己生成 ABC →
将真实 ABC 交给 `cot=full`。使用缓存中的完整《今晚不眠》歌词／风格、seed12300，
音频帧预算9000（25 fps，最多约6分钟），不是要求模型唱满6分钟。
`--mode off` 只建立无规划基线；已经完成此步骤后，`--mode full` 只做规划和完整生成。

从本目录运行，自动读取现有 plist 选择实际服务的私有解释器：

```bash
python="$(python3 -c 'import pathlib,plistlib; p=pathlib.Path.home()/"Library/Application Support/yue2-3b/com.local.yue2-3b-metal.plist"; print(pathlib.Path(plistlib.loads(p.read_bytes())["ProgramArguments"][0]).with_name("python"))')"
"$python" -I establish_baseline.py --mode all \
  --output "$HOME/Library/Application Support/yue2-3b/baselines/$(date +%Y%m%d-%H%M%S)"
```

规划沿用原生 ABC preset（4096 token 上限、32 token 下限）；下限通过
`yue2_min_tokens` 交给模型自己的 sampler，不设置 Metal 不支持的 vLLM 顶层
`min_tokens`。客户端只用 checkpoint 的 `qwen.tiktoken` 解码返回 token IDs，
要求真实 `ABC_END` 终止证据，不接受截断计划、不补结束符、不使用手写或旧 ABC 代替。
随后完整音频必须 HTTP200、`X-Audio-Truncated:false`，且可解码为有限、非静音、
合法幅度的48kHz双声道 WAV；任何阶段失败即保留证据并停止，不退回短样本。

每次请求采用既有生产 HTTP 1800秒上限，监控 available RAM≥4GiB、
服务进程树 RSS≤14GiB、磁盘余量≥2GiB。超过门槛只断开本次请求的 socket；
未取得服务端取消确认时不声称已取消推理。证据包括实际服务配置、checkpoint／选定源码
指纹、原输入、完整请求／响应头、规划 token IDs／ABC、WAV、终止与资源记录。
RSS 不是 MPS 显存，客户端的 MPS 指标也不是服务指标。

完整自然结束和客观音频通过不等于歌词唱完或音质合格，仍需人工试听。
缓存配套 MP3 标注为 **earlier checkpoint**，不是当前3B权重的位等价参考或强制目标时长。
规划与无规划的工作量、输出长度不同，单次耗时不能用作量化加速或同工作量 A/B。

本次真实样本（同一完整歌词／风格、seed12300，各一路一条；不是速度 A/B）：

| 模式 | ABC规划 s | 音频生成HTTP s | 输出音频 s | 原生结束 |
| --- | ---: | ---: | ---: | --- |
| 原精度 `off` | — | 547.431 | 204.759 | 未截断 |
| 原精度 `full` | 50.710 | 510.820 | 182.839 | 未截断 |

两路均为48kHz双声道PCM16、有限非静音、无越界幅度；full两次HTTP合计561.530秒，
整条CLI569.16秒（含元数据／指纹检查）。full请求期间最低available RAM8.558GiB，
进程树RSS最高1.273GiB；没有服务侧MPS显存峰值数据。继续执行时原任务已经正常关闭，
仅通过现有启动脚本启动YuE2原精度服务，未停止其他模型，未迁移共享后端；服务继续运行。

ABC确实由当前模型生成：2006个文本token加真实 `ABC_END`，`finish_reason=stop`、
`stop_reason=151848`。**但结构检查未通过**：`score.abc` 第3行是 `ttpM:4/4`，
第11行混入 `_TOOLTIP`，第81／102行存在未闭合和弦引号，其他乐句也有无关字符。
原样保留并用于本次full生成，没有清洗、补写或替换。不能将本次full称为质量合格、
与vpipe听感已对齐或歌词完整；两路均未人工听评。

汇总在相邻后端 `outputs/yue2-original-full-20261017/summary.json`；试听产物：
`off/original-off.wav`、`aligned-full-native-min/full/audio.wav`，ABC在
`aligned-full-native-min/planning/score.abc`。首个顶层 `min_tokens` 请求的HTTP400也保留在
`aligned-full/`，不是成功样本。新增基线／终止边界回归54项通过，两工具和两测试文件
Ruff检查／格式检查通过，私有解释器 `-I` CLI及完整HTTP路径已实际运行。
本次没有通用后端／依赖／entry point／native资源变更，不改安装器、Formula或发布远端。

## 历史检查和限制

```bash
bash -n start.sh stop.sh
docker compose -f ../docker-compose.yml config --quiet
# Metal 私有环境自检见上面的 base/runtime 命令（无需下载模型）。
```

已构建私有 patched Omni，`pip check` 与私有 CLI 通过，真实 MPS 自检通过。Shell 语法与原独立代理 Compose 配置曾检查通过；现使用根目录总代理配置。

此前服务启动验证曾通过真实权重加载、launchctl、8096及原独立代理的health／模型身份检查。原代理已移除，80端口由根目录总代理管理；当时没有调用服务启停脚本或停止其他任务。以下为2026-10-15短样本历史记录，不替代上文完整歌曲证据；并发／服务端取消、长热负载、人工听感及歌词准确率仍未验收。

本轮实际使用当前源码和本地真实权重，通过 `AsyncOmni` 完成 AR→NAR→VAE attention A/B；仅资源参数改为 context6144、KV1GiB，保留输入、采样和拓扑。BF16模型、FP32 VAE、seed42、512帧预算；各路独立warmup，再正反交错各计时两次：

| Attention 路径 | 两次完整生成 s | 中位 s |
| --- | --- | ---: |
| MLX bridge | 54.953 / 61.322 | 58.138 |
| Torch stream | 41.130 / 41.079 | 41.105 |

本次耗时比1.414×、延迟降低29.30%；仅此输入/设备的两次样本，不外推其他负载。输出20.4787秒、48kHz双声道，有限非静音；各路重复及跨路浮点音频完全一致，每请求14364次AR paged attention命中已核对。ANE明确关闭，当时通用 elementwise AOT产物不存在，不能将收益归因于ANE或新融合shader，也不代表当前命名迁移后的产物状态。原始记录/WAV在相邻后端的 `outputs/yue2-real-current-20261015-204628/`。

11项CPU行为回归通过；真实ANE拒绝复验在prepare立即退出且正常清理，没有继续使用关闭组生成。环境元数据将TorchVersion转换为普通字符串后通过安全msgpack编码，不启用不安全pickle。本次GPU及拒绝运行的进程均已退出。

权重许可为 **CC BY-NC 4.0**；本目录和后端的软件许可不赋予权重商业使用权。
