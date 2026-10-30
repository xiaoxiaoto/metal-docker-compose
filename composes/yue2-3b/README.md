# YuE2-3B · vLLM-Omni / Omni Metal 部署

参考 [模型卡](https://huggingface.co/m-a-p/YuE2-3B) 和 `../higgs-tts-3/`。独立端口、服务和缓存，不修改 Higgs / H2O 或通用 Metal 后端。

量化默认 `off`，仅接受 `off/dynamic`；dynamic 必须使用完整离线搜索批准的私有 profile，且 ANE off。旧 `manual-w8a16` 启动接口和手动授权不能用于 dynamic。
根目录将本模型配置为 **dynamic，空闲300秒卸载**。每次根目录启动（包括默认/gateway/lifecycle）先安全停止所有依赖模型、删除历史共享环境和私有 overlay，再重建共享后端；停止失败不删除，构建失败不回退。`./start.sh yue2` 每次调用本目录 `./start.sh --prepare-only` 重建私有环境和 plist，首次推理才离线加载。权重、下载缓存、量化资产和日志保留；精度/量化参数以本次设置为准。详见[根目录重建合同](../../README.md#每次根目录启动前重建共享环境)。

冷状态的聚合列表不含本模型，单模型身份查询返回503且不唤醒；先在根目录查看 `python3 -I runtime_mgmt/lifecycle_control.py status`，发送下文真实音频请求，完成后再检查身份。首次请求在热推理预算之外另留最多600秒加载时间。动态请求只走统一入口，不直连8096；完整音频/流结束前不会启动空闲计时。

## 原生拓扑和版本

- YuE2 是单阶段 `AR（semantic tokens）→ terminal NAR flow matching → VAE`，不是 Higgs 的 Talker → Code2Wav 双阶段，不能复用其 connector 或 codec 配置。
- 共享依赖仅由根目录 [`pyproject.toml`](../../pyproject.toml) / [`uv.lock`](../../uv.lock) 声明，使用 `uv sync --locked` 非 editable 安装：vLLM / vllm-metal `0.32.0`，Omni 固定 `9842ebc17cf1d9c6b440ecb6a79039162f68a1dd`，Omni Metal 来自 `https://github.com/xiaoxiaoto/vllm-omni-metal.git` 的 `5774df7d720b3a8c6d32035729b42c31e480f346`。共享准备不依赖相邻源码；官方联网预检不决定新 commit、不自动升级 main，MPS 与 `pip check` 等验证通过后才激活。每次根目录启动都重建共享环境，私有准备器仍保留自身源码流程与来源兼容门禁。不要混用历史 overlay 或桥接模块布局；详见[根目录合同](../../README.md)。
- 私有准备器要求基础环境中 Omni 的 `direct_url.json` 指向官方 Git 仓库且包含完整 commit；来源或元数据不匹配、补丁不能应用时拒绝准备，不绕过检查。
- vLLM 与 vllm-metal 也从官方 GitHub 源码安装，分别固定 `vllm-project/vllm` 的 `5df63a7203cdb9328d824a27b682e77c7b9928be` 与 `vllm-project/vllm-metal` 的 `6d6f2c37f8242f1ff1a336534adf6c2d2fe2936d`（v0.32.0）；由 uv 隔离构建非 editable wheel，不使用官方预编译 wheel。vLLM 的 CPU 编译形态只是 Metal 插件底座，不代表模型 CPU fallback。
- checkpoint 的架构名是 `YuE2ForCausalLM`，profile 用 `hf_overrides.architectures: [Yue2ForCausalLM]` 选择 Omni 原生类，不执行 checkpoint remote code。
- YuE2 的实际文本 tokenizer 是本地 `qwen.tiktoken`，engine 接收 token IDs 并设置 `skip_tokenizer_init: true`。本 profile 显式用 `tokenizer: null` 覆盖上游无用的 `gpt2` 占位引用，避免 vLLM 在离线模式下仍尝试解析 GPT-2 缓存；不额外下载 GPT-2，也不关闭离线限制。
- 同时需要未量化 `m-a-p/YuE2-3B`（AR/NAR BF16，约 7.26 GB）和标准 `m-a-p/YuE2-Vae`（FP32，约 0.53 GB）；输出目标是 **48 kHz 双声道**。默认使用 ModelScope 完整下载并保留仓库文件/缓存元数据，不安装仓库内的推理 wheel；显式选择 HF 时只下载模型必需文件。
- 单请求默认 profile；Metal 使用 CUSTOM MPS SDPA、同步 uni 和关闭 prefix cache；保留 `max_model_len=24576`，KV恢复量化前的 `gpu_memory_utilization=0.30` 自动预算，不再固定3GiB。NAR/VAE 在 KV 预算之外分配，不能把该比例视为总内存上限；实际 bootstrap 前至少16GiB available、部署保留2GiB磁盘，不提高 MPS 水位。`--prepare-only` 不受真实模型加载的16GiB门槛限制。

## Apple Silicon / Metal

需要 Apple Silicon、macOS 15+、Homebrew、Python 3.11+（根控制脚本与启动引导）、ModelScope CLI；uv 托管的共享运行时要求 Python `>=3.12,<3.13`。根目录总代理另需 Docker Compose。与 Higgs 相同：模型在 macOS launchctl 中运行，本目录不再运行 Docker 代理，不能在 Linux 容器中使用 Metal。

`../../extensions/vllm_omni/yue2_3b/yue2-mps.patch` 对固定 Omni 源码进行小范围修复，保留原生采样、NAR/VAE 算法及错误处理：

1. 请求 generator 使用输入张量的设备，MPS 张量不再配 CPU generator。
2. 只有 CUDA 使用 pinned memory / event；MPS 同步读回，仍保留上游的延迟接受 token、重算与结束语义。
3. VAE 初始化和同步解码跟随模型参数的设备，MPS 解码保持 FP32，最终音频搬回 CPU。

`prepare_runtime.py` 从共享环境记录的官方 commit 构建 **YuE2 私有 Omni wheel**，安装到 `~/.local/share/local-model-backend/overlays/yue2-3b/<fingerprint>/`，版本标记为 `0.32.0.dev0+cpu.yue2mps2`。不修改共享后端已安装的 Omni 文件、通用 Metal 后端或模型 registry，不复制第二套模型实现，不模拟 CUDA、不启用 CPU fallback。

私有环境覆盖 Omni；显式 `VLLM_OMNI_METAL_WHEEL=/absolute/backend.whl` 还会仅在该 overlay 安装通用后端 wheel，Torch / vLLM / vllm-metal 依赖仍通过 `.pth` 复用冻结基础环境，不修改共享后端或 `current`。指纹包含补丁、固定 commit、基础路径、Python版本、准备器及 `extensions/vllm_omni/yue2_3b/` 下全部五个私有模块、共享后端源码/native资源及所选 wheel 的路径与内容。生产私有模块通过 Omni wheel 的 `py-modules` 正常分发，不在安装后复制进 site-packages；ready 在选定 Python `-I` 下检查每个模块源码 SHA、overlay 安装归属和 RECORD、通用 `quantization.engine/linear` 导入和实际 Torch-stream launcher。已有 ready 也重新检查，失败撤销 ready。所选 wheel 路径保存为状态目录中的 `backend-wheel`；已选择但丢失时拒绝启动，不静默退回旧包。旧 overlay 保留，不在引擎启动/请求期间 JIT。

新私有环境不运行 ensurepip，pip 与其他共享依赖一样通过基础环境 `.pth` 继承，避免 venv 自带的旧 pip 遮蔽共享版本；准备时仍原样执行 `pip check`。准备器本身参与指纹，因此此变更会创建新的 overlay，不修改或删除旧 overlay（包括内含旧 pip 的环境）。

通用后端 GPU 算子位于 `vllm_omni_metal.ops`，ANE 运行时为 `vllm_omni_metal.ane.runtime`，离线构建入口为 `vllm_omni_metal.ane.build`；准备器与 ready 校验使用这些路径，源码和资源指纹递归包含子包。

## 启停

只准备、不立即加载：

```bash
cd composes/yue2-3b
./start.sh --prepare-only
```

准备模式执行环境准备、版本检查、`check_metal.py` tiny MPS 自检、模型下载/验证及 plist 生成，通过 `plutil -lint` 后退出，不 bootstrap 或加载真实模型。16GiB available 仅在实际 bootstrap 前检查，2GiB磁盘门槛保留。已有注册任务或 8096 冲突仍拒绝；未提供 `VLLM_BACKEND_PYTHON` 时调用共享准备器，安全停用全部依赖模型并删除历史环境后重建。非法参数非零退出；无参数仍准备后立即加载：

```bash
cd composes/yue2-3b
./start.sh  # 自检通过后准备 ModelScope 模型并加载本机模型；总代理在根目录单独启动
./stop.sh
```

`start.sh` 未收到 `VLLM_BACKEND_PYTHON` 时调用根目录共享准备器，联网预检后安全停止所有依赖模型、删除历史环境并重建，再准备 YuE2 私有 patched Omni overlay。不修改 Homebrew Formula。

根目录 `./start.sh yue2` 每次重新准备私有 overlay 和 plist 后保持冷状态，不再复用旧 plist。准备只发生在显式启动阶段；HTTP 冷启动复用本次已准备部署，不下载、不重建环境。

`check_metal.py` 不再依靠源码字符串判断：实际执行上游 MPS 请求 RNG、完整 sampler / 同步读回、同 seed 与合批隔离、帧上限结束、请求清理、非法分数拒绝，以及随机 tiny NAR 和 FP32 MPS VAE 解码/有限值检查；另检查缺少绑定产物时通用 DenseSwiGLU 保留原生 GPU FFN。它不下载或加载真实大权重，也不是听感验收。通过检查或 `/health` 不等于实际歌曲生成成功。

### 显式离线准备 GPU + ANE

YuE2 私有补丁在完整 NAR 结构后接入通用 `DenseSwiGLU`。生产启动固定旧后端 ANE 为 off，不接受旧变量覆盖；当前 Omni 尚未实现公共 `ane` profile 执行合同，选择 `ane` 明确拒绝，不能把旧私有 profile 作为公共批准。以下命令仅为隔离离线诊断，普通部署不会编译工具或下载模型。

使用已缓存真实权重执行一次 256–512 帧离线校准（启动前可用 RAM ≥16 GiB，运行时可用 RAM ≥4 GiB、整个子进程树 RSS ≤14 GiB、阶段≤900 秒、请求≤180 秒、磁盘保留≥2 GiB）：

```bash
/实际路径/backend-python -m vllm_omni_metal.ane.build --bridge
VLLM_PLUGINS=metal,vllm_omni_register_models /实际路径/yue2-private-python ../../extensions/vllm_omni/yue2_3b/prepare_ane.py --frames 512
```

校准通过新的隔离 worker 完成，不访问或重启 8096/launchd，也不修改默认服务资源。`--deploy-config /path/to/profile.yaml` 可选择经过独立内存测量的资源 profile；普通ANE校准仅注入观测扩展，不改请求、拓扑、attention 或资源值，只有显式 GPU attention A/B 会切换被测路由。使用源码后端时显式传入相应 `PYTHONPATH`，私有 overlay 仍须使用 `yue2mps2` 版本。

共享 RPC 严格要求 `{"supported": true, "result": {...}}`；上游把异常转换成的 `{"supported": false, "error": ...}` 会原样写入 `rpc-error.json` 并终止，**不继续生成、不自动切成 GPU 测量、不假装 ANE 成功**。ANE 准备/构造失败时保留原健康 GPU FFN 组；只有请求已清理且候选 pool/disabled/closed 验证通过后才替换并关闭旧组。批准后的默认组也采用事务替换。

证据保存在系统临时目录 `yue2-ane-*`（macOS 通常位于 `/var/folders/.../T/`）。ANE 校准检查同模式同 seed 音频可复现，并要求整请求候选稳定快至少 3% 才批准默认比例；FP16 ANE 不代表 BF16 等价或无损，需另行听测。准备时间和图磁盘另计，短音频的行数可能不足以协同。

### 显式 GPU-only 真实测量与 attention A/B

ANE 被拒绝后，需**另起显式命令**测量 GPU，而不是在失败流程内自动继续：

```bash
VLLM_PLUGINS=metal,vllm_omni_register_models /实际路径/yue2-private-python ../../extensions/vllm_omni/yue2_3b/prepare_ane.py \
  --gpu-only --frames 512 --deploy-config /path/to/profile.yaml

# 仅 GPU-only 合法：同 worker 比较旧 MLX bridge 与当前 Torch-stream launcher
VLLM_PLUGINS=metal,vllm_omni_register_models /实际路径/yue2-private-python ../../extensions/vllm_omni/yue2_3b/prepare_ane.py \
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
  /实际路径/yue2-private-python ../../extensions/vllm_omni/yue2_3b/prepare_quantization.py \
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

### 显式固定 W4/W8/BF16 混合精度诊断（不批准）

从通用后端 checkout 根目录运行已有私有解释器，`--precisions` 显式指定最多四个
不同 NAR 独立 down 投影，并且至少各有一个 W4A16、W8A16 和保留 BF16：

```bash
PYTHONPATH="$PWD" VLLM_PLUGINS=metal,vllm_omni_register_models \
  METAL_MODEL_OPTIMIZERS= METAL_QUANTIZATION=off /实际路径/private-python \
  ../../extensions/vllm_omni/yue2_3b/prepare_quantization.py \
  --precisions '{"nar_layers.0.mlp.down_proj":"w4a16","nar_layers.9.mlp.down_proj":"w8a16","nar_layers.18.mlp.down_proj":"bf16"}' \
  --frames 512 --deploy-config /实际路径/配置.yaml --output /实际路径/空证据目录
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
权重、不重启服务或修改 ready／plist／部署默认。上述命令仅用于诊断。


### 生产部署：离线批准的 dynamic profile（默认关闭）

`start.sh` 默认 provider 列表为空且量化 off；启用须显式设置
`METAL_MODEL_OPTIMIZERS=dynamic_quantization`、`METAL_QUANTIZATION=dynamic` 和
绝对路径 `METAL_QUANTIZATION_PROFILE`。当前 Omni 拒绝 `ane`、`streaming`、
`gpu_ops`，不会把基础 Metal provider 当作本执行器支持。共享
`load_profile(path, *, scope)` 支持单个schema2或 `{"profiles": {实际完整scope: profile}}`
集合；文件必须当前用户owned0600、非符号链接、4MiB内普通文件，dynamic强制ANE off。
完整scope、权重、实现指纹及批准证据仍由生产适配器验证；不能手改approved或
用历史manual policy绕过完整批准，缺失/过期/不支持合同明确报错，不静默fallback。
通用 `vllm_omni_metal.quantization.engine.DynamicEngine` 执行搜索／应用，
模型私有模块只提供 NAR 适配与安装验证；请求期间不搜索、不校准、不写 profile。
五个模型私有模块均作为 Omni wheel 的 `py-modules` 安装在同一目录；运行时 worker extension 与运行时量化适配器均从该 overlay 导入。动态搜索准备器也随 wheel 分发，仅供离线准备及其源码指纹验证，不在生产启动时导入；ready 检查所有模块的 SHA 与 RECORD 归属。
ready同时导入通用 `quantization.capture`、`adapter`、`runtime` 新API，
所有安装校验使用选定Python `-I`。FP16/BF16保持源dtype，FP32只identity；
通用入口接入不代表YuE2真实动态质量/速度通过，不自动启用或迁移现有环境。

先用所选私有解释器进行完整真实模型离线敏感度搜索；`--output` 必须是空目录：

```bash
PYTHONPATH=/absolute/vllm-omni-metal VLLM_PLUGINS=metal,vllm_omni_register_models \
  METAL_MODEL_OPTIMIZERS= METAL_QUANTIZATION=off /实际路径/yue2-private-python \
  ../../extensions/vllm_omni/yue2_3b/prepare_dynamic_quantization.py --frames 512 --candidate-layers auto \
  --output /absolute/empty-dynamic-evidence
```

`candidate-profile.json` 是候选而非部署授权；只有全部完整批准条件通过才输出
`approved-profile.json`。质量／速度／释放浮点后的真实生成等任何门槛失败，
保留拒绝证据，不发布生产 profile。不承诺无损或加速。

在已获准管理该模型、且服务已停止时，显式指定已有基础解释器和新后端 wheel：

```bash
BASE=/absolute/existing/base/bin/python
VLLM_BACKEND_PYTHON="$BASE" VLLM_OMNI_METAL_WHEEL=/absolute/backend.whl \
  METAL_MODEL_OPTIMIZERS=dynamic_quantization METAL_QUANTIZATION=dynamic \
  METAL_QUANTIZATION_PROFILE=/absolute/empty-dynamic-evidence/approved-profile.json \
  ./start.sh
```

首次成功准备后保存 wheel 选择，下次仍应提供 `VLLM_BACKEND_PYTHON`。
profile 原子复制到 TCC-safe 状态目录，launchd 不依赖 Documents 或 PYTHONPATH。
已注册任务／占用端口仍明确拒绝；随机 tiny 自检强制 quantization=off，不是量化生产命中。

只检查私有环境，不下载模型或更新 Homebrew：

```bash
base="/绝对路径/已核对基础环境/bin/python"
runtime="$("$base" -I prepare_runtime.py)"
VLLM_PLUGINS=metal,vllm_omni_register_models "$runtime/bin/python" -I check_metal.py
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
  'http://127.0.0.1/v1/models'

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

默认关闭量化；AR/NAR 保留 BF16、VAE 保留 FP32、ANE off。完整歌曲仍需真实生成、自然结束和人工试听验收。

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


权重许可为 **CC BY-NC 4.0**；本目录和后端的软件许可不赋予权重商业使用权。
