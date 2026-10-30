# SeedVR2 7B FP16 · Omni + Metal

本目录提供 Apple Silicon 单 MPS 的原生 `SeedVR2Pipeline` 部署代码，规范模型 ID 为 `seedvr2`，提供图片超分与短视频修复入口。健康检查成功不能替代真实推理验收。
根目录将本模型配置为 **dynamic，空闲300秒卸载**。每次根目录启动（包括默认/gateway/lifecycle）先安全停止所有依赖模型、删除历史共享环境和私有 overlay，再重建共享后端；停止失败不删除，构建失败不回退。`./start.sh seedvr2` 每次调用本目录 `start.sh --prepare-only` 重建私有环境和 plist，首次推理才离线加载。权重、下载缓存、量化资产和日志保留；FP16/量化参数以本次设置为准，准备成功不等于真实推理通过。详见[根目录重建合同](../../README.md#每次根目录启动前重建共享环境)。

冷状态的聚合列表不含本模型，单模型查询返回503且不唤醒；先查看根目录 `python3 -I runtime_mgmt/lifecycle_control.py status`，发送下文真实图片/视频curl，完成后复查身份。冷请求在热推理预算之外另留最多1800秒加载预算；不旁路8100推理。动态启动额外要求至少24GiB保守空闲内存，不提高MPS水位。


## 固定模型与安全边界

- 来源 [Comfy-Org/SeedVR2](https://modelscope.cn/models/Comfy-Org/SeedVR2)，固定 revision `5360e873f15685f17acdb077b5e46d16eb6d05d7`；仅普通 7B FP16 checkpoint，不支持 sharp、预量化 checkpoint 或 3B。独立 DynamicEngine 仅消费真实批准 profile，不解除原生 config 量化拒绝。原始模型卡为 [ByteDance-Seed/SeedVR2-7B](https://huggingface.co/ByteDance-Seed/SeedVR2-7B)。
- 只下载 `diffusion_models/seedvr2_7b_fp16.safetensors`（16480583960 字节，SHA256 `2742ca6fee63bc5cc1773f426dd4b07b78cad27f51c9ea5cd42b035e6b592252`）和 `vae/ema_vae_fp16.safetensors`（501324814 字节，SHA256 `20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1`）。
- 保留作者缓存路径 `~/.cache/modelscope/hub/models/Comfy-Org/SeedVR2`。完整缓存流式复核 SHA256 后复用；大小正确但校验失败的文件直接报错，不静默覆盖。下载前要求完整权重加 5 GiB 可用磁盘，不自动删除缓存。
- 不执行缓存 Python、不反序列化 pickle、不加载真实张量。检查路径边界、safetensors header、dtype、shape、offset、精确文件大小和固定发布哈希。`model_index.json` 原子创建为 `{"_class_name":"SeedVR2Pipeline","seedvr2_variant":"7b-comfy-fp16"}`；既有内容不同则拒绝，不覆盖用户配置。
- 模型专属适配仅在私有 Omni overlay wheel；不修改共享已安装包、不覆盖 registry、不修改相邻 `vllm-omni-metal`，没有独立 HTTP adapter。
- 适配核对官方 SeedVR 源码 commit `e4de8c24441a67e1b7df56abea10645059bb1185`（`configs_7b/main.yaml`），Omni 基线为实际 0.32 源码 `9534efe802ef79ce52af31847fb144f63d259e90`，私有 wheel 版本为 `0.32.0.dev0+cpu.seedvr2mps2`。7B 保留末层 text 分支，使用带 bias 的 GELU(tanh) MLP、仅视频 pixel RoPE，没有输出 RMSNorm/OutAda；不能把未适配的 3B base 当成 7B pipeline。
- 参考 [Omni SeedVR2 文档](https://github.com/vllm-project/vllm-omni/blob/9534efe802ef79ce52af31847fb144f63d259e90/docs/models/seedvr2.md)和[官方 7B 配置](https://github.com/ByteDance-Seed/SeedVR/blob/e4de8c24441a67e1b7df56abea10645059bb1185/configs_7b/main.yaml)。Omni 基线仅有 3B，本目录私有补丁扩展 7B。**私有准备器要求 Omni `9534efe802ef79ce52af31847fb144f63d259e90`，与共享锁定的 `9842ebc17cf1d9c6b440ecb6a79039162f68a1dd` 不匹配，当前启动和校准受阻**；须先适配固定共享来源，不得绕过门禁或用未打补丁的 base CLI 启动 7B。

## 环境、启动和停止

需要 Apple Silicon macOS 15+、MPS、Python 3.11+（根控制脚本与启动引导）、ModelScope CLI，以及根目录共享后端/总代理要求的工具。共享运行时由 uv 托管，要求 Python `>=3.12,<3.13`；共享依赖仅由根目录 [`pyproject.toml`](../../pyproject.toml) / [`uv.lock`](../../uv.lock) 声明，以 `uv sync --locked` 非 editable 安装：vLLM/Metal `0.32.0`，Omni 固定 `9842ebc17cf1d9c6b440ecb6a79039162f68a1dd`，Omni Metal 来自 `https://github.com/xiaoxiaoto/vllm-omni-metal.git` 的 `5774df7d720b3a8c6d32035729b42c31e480f346`。共享准备不依赖相邻源码，保留官方联网预检但不随 main 自动升级，MPS 与 `pip check` 等验证通过后才激活；模型私有 overlay 的源码与基线门禁流程不变，详见[根目录合同](../../README.md)。实际执行以下根入口命令会在缺少 plist 时准备软件和权重，模型仍等待首次推理才加载；仅阅读代码无需执行。

vLLM 与 vllm-metal 也从官方 GitHub 源码安装，分别固定 `vllm-project/vllm` 的 `5df63a7203cdb9328d824a27b682e77c7b9928be` 与 `vllm-project/vllm-metal` 的 `6d6f2c37f8242f1ff1a336534adf6c2d2fe2936d`（v0.32.0）；由 uv 隔离构建非 editable wheel，不使用官方预编译 wheel。vLLM 的 CPU 编译形态只是 Metal 插件底座，不代表模型 CPU fallback。

```bash
# 首次更新代理路由：会中断在途请求，先确认可重载
./start.sh gateway
./start.sh seedvr2
# 只停止本模型；不停止总代理或其他模型
./stop.sh seedvr2
```

可单独执行 `composes/seedvr2/start.sh --prepare-only`，优化仅通过公共 `METAL_*` 环境变量配置。准备模式复用完整流程，在原子写入 plist 并通过 `plutil -lint` 后退出，不 bootstrap 或停止已有 worker；已注册任务和端口冲突仍拒绝，非法参数非零退出。无参数直接运行本目录 `start.sh` 仍准备并加载模型。

`start.sh` 先校验共享 `VLLM_BACKEND_PYTHON` 和资源，再准备私有 runtime 与权重；CLI 使用私有 `overlay/bin/vllm-omni`，但 `VLLM_BACKEND_PYTHON` 仍指向共享基础环境。launchctl label 为 `com.local.seedvr2-metal`，监听 `127.0.0.1:8100`；plist 位于 `~/Library/Application Support/seedvr2/`，日志为 `~/Library/Logs/seedvr2-metal.log` 和 `seedvr2-metal.error.log`。不是登录自启项。

未提供 `VLLM_BACKEND_PYTHON` 时，本目录启动脚本调用共享准备器，联网预检后安全停用全部依赖模型、删除历史环境并重建。`--prepare-only` 本身不 bootstrap 或加载真实模型，不接管已注册的本模型任务；停止失败或残留 worker 会阻止环境删除。

私有部署来源指纹包含固定官方 commit、私有版本、补丁、构建器、模型资源及当前后端源码。复用或新建均检查私有包路径、版本、资源 SHA256 和 wheel RECORD，并运行 `pip check` 后才接受 ready。量化和 ANE 默认关闭。
新建 overlay 不运行 ensurepip；pip 与基础依赖通过明确的共享 base `.pth` 继承。准备器变化会产生新指纹，不复用旧私有 pip 环境。


准备及显式加载的设备能力预检保守要求：权重总字节数加 **10 GiB RAM**，以及权重加 **4 GiB Metal 推荐预算**；`--prepare-only` 不跳过这些检查，下载磁盘检查也保留。这只是预检门槛，**不是测得峰值或可运行保证**。权重和矩阵投影固定 float16，7B 文本分支门控输出与残差保留 float32，归一化后在进入投影前转回 float16；不是全模型升精度。保持 eager、单 GPU、uni executor、TORCH_SDPA 和 VAE tiling；不启用 slicing、offload、原生 config 量化或 MPS fallback，不调整内存水位。动态量化默认 off，仅显式 dynamic 消费批准文件。启动后离线加载、spawn worker，最多等待 1800 秒；失败仅清理本次成功注册的任务。已有注册任务或占用端口导致明确失败，不杀进程。

### 7B 图片数值边界

私有补丁对 NormalMLP 的 FP16 文本路径保留 FP32 门控输出和残差，直到下一次归一化；矩阵输入和模型参数仍为 FP16，视频分支、BF16 路径及原 3B SwiGLU 路径不因此改动。不采用 clamp、nan_to_num、CPU fallback 或整模型升精度，也不宣称与官方 CUDA BF16 autocast 逐位等价。精度边界变化后须重新校准，不能复用旧批准。


## 身份、图片超分与短视频验收

模型加载后先检查无参数聚合身份，再检查单模型身份；冷状态先查看生命周期状态并完成首次真实推理。统一入口只面向本机、无鉴权，勿暴露公网。

```bash
curl --fail-with-body --show-error --connect-timeout 5 --max-time 10 \
  'http://127.0.0.1/v1/models'
```

### 图片超分

使用 Omni 原生 `/v1/images/edits`，上传单张 PNG，不使用文生图接口。将 `/实际路径/input.png` 替换为存在的图片；例如输入 256×256，指定输出 `size=512x512` 就是 2 倍超分。`size` 是输出宽×高，不是倍率参数，两个维度须为 16 的倍数。输入与输出每张均不得超过 407,040 像素，整个 multipart 不得超过 25 MiB。

```bash
set -o pipefail
curl --fail-with-body --show-error --connect-timeout 5 --max-time 10 \
  --get 'http://127.0.0.1/v1/models' \
  --data-urlencode 'model=seedvr2'

curl --fail-with-body --show-error --connect-timeout 5 --max-time 1800 \
  'http://127.0.0.1/v1/images/edits' \
  --form 'model=seedvr2' \
  --form-string 'prompt= ' \
  --form 'image=@/实际路径/input.png;type=image/png' \
  --form 'size=512x512' \
  --form 'n=1' \
  --form 'num_inference_steps=1' \
  --form 'guidance_scale=1' \
  --form 'seed=7723' \
  --form 'response_format=b64_json' \
  --form 'output_format=png' \
  --dump-header /tmp/seedvr2-image-test.headers | \
  python3 -c 'import base64,json,sys; from pathlib import Path; Path("/tmp/seedvr2-image-test.png").write_bytes(base64.b64decode(json.load(sys.stdin)["data"][0]["b64_json"], validate=True))' &&
open /tmp/seedvr2-image-test.png
```

此编辑接口只支持 JSON `b64_json`，不是 `response_format=file`；管道从 JSON 解码 PNG，不保存 JSON 结果文件。响应头在 `/tmp/seedvr2-image-test.headers`，Content-Type 应为 JSON；图片在 `/tmp/seedvr2-image-test.png`。curl／解码失败不打开旧文件；成功需 HTTP 200、可解码的 512×512 有效 RGB PNG，并与输入比对细节、颜色及伪影。单图按单帧运行同一 VAE／DiT／颜色校正，返回 PNG 而非单帧 MP4；不支持文本编辑、mask、多图或透明通道保留，实际超分效果未验收。

### 短视频修复

将 `/实际路径/short.mp4` 替换为存在的短 CFR 视频，首次建议仅 5 帧；整个 multipart 请求（含边界和字段）不得超过 25 MiB。

```bash
curl --fail-with-body --show-error --connect-timeout 5 --max-time 10 \
  --get 'http://127.0.0.1/v1/models' \
  --data-urlencode 'model=seedvr2'

curl --fail-with-body --show-error --connect-timeout 5 --max-time 1800 \
  'http://127.0.0.1/v1/videos/sync' \
  --form 'model=seedvr2' \
  --form-string 'prompt= ' \
  --form 'input_references=@/实际路径/short.mp4;type=video/mp4' \
  --form 'size=224x128' \
  --form 'num_inference_steps=1' \
  --form 'guidance_scale=1' \
  --form 'seed=7723' \
  --dump-header /tmp/seedvr2-test.headers \
  --output /tmp/seedvr2-test.mp4 &&
open /tmp/seedvr2-test.mp4
```

`prompt` 必须是单个空格；不传 `fps`，保留输入源 fps。输入使用短 CFR 视频，输出宽高必须为 16 的倍数。视频仅支持 `/v1/videos/sync` 同步短视频；长视频 sharded route 在本单 MPS 部署不支持，不使用 `restore-long`，也不将分片长视频伪装为已支持。响应头与 MP4 保存于 `/tmp/seedvr2-test.headers`、`/tmp/seedvr2-test.mp4`；复测覆盖，不能把 JSON 错误或旧文件当成功。检查 HTTP 200、视频 Content-Type、可解码 MP4、帧数/尺寸/fps/有效画面，并对比输入；HTTP 成功仍不等于画质或性能达标。

原生 seed-only 请求使用 MPS RNG，显式 CPU 噪声仍受支持；必须使用 `--form-string 'prompt= '` 保留单空格，普通 `--form` 会去除尾随空白。


## 无副作用检查

`composes/seedvr2/start.sh --help`、`composes/seedvr2/stop.sh --help`、`python3 composes/seedvr2/prepare.py --help` 无部署副作用。`prepare.py --check-only` 仅校验现有缓存，不下载、不写文件（完整真实缓存哈希读取会较慢）。使用前核对模型许可证与数据使用约束。

## DynamicEngine：真实批准后才开启

`seedvr2mps2` 私有 wheel 将 `extensions/vllm_omni/seedvr2/` 中的 `quantization.py`、`joint.py`、安全权重清单和完整图校准器一同打包；准备器来源指纹和 `.ready` 检查绑定各文件 SHA256。只在实际 `load_weights` 完成后替换选中投影，不修改已安装上游，不注册第二套模型；请求阶段不读 profile、不校准、不编译、不构造权重。
私有源码固定为实际0.32官方提交 `9534efe802ef79ce52af31847fb144f63d259e90`；准备器从相邻后端 checkout 的干净副本构建非 editable 后端 wheel，只安装到本模型私有 overlay。指纹覆盖 `pyproject.toml`、完整 Python/native 资源、README 和许可证；ready 校验私有导入路径、随包文件哈希、许可证及安装 RECORD。共享 base/current 不被修改；`.pth` 只复用基础依赖，不冒充完整依赖重装。
安装/ready校验使用选定Python `-I` 并导入通用 `quantization.capture`、
`adapter`、`runtime`。源FP16/BF16保持dtype，通用FP32只identity；
这些安装检查不替代本模型真实7B整图质量与速度门禁，不自动启用或发布。
动态profile路径必须为绝对路径；相对路径明确拒绝。

候选仅为 36 个 block 中第 0/12/24/35 层的 `mlp.vid.proj_out`，最多批准其中一个。7B 保持带 bias 的 `Linear → GELU(approximate="tanh") → Linear`，没有 SwiGLU/gate-up 假设；text、QKV、VAE、其它线性层不量化。普通投影返回 Tensor，拒绝 vLLM tuple/skip-bias Linear 子类，不伪造 tuple 适配。应用前检查完整模块树的 module、parameter、buffer alias；全部校验和量化构造成功才提交，失败保留原图。

`METAL_QUANTIZATION=off|dynamic` 默认 `off`；dynamic 必须同时设置 `METAL_MODEL_OPTIMIZERS=dynamic_quantization` 与绝对路径 `METAL_QUANTIZATION_PROFILE`。模型私有变量和量化启动 flags 不再使用。本 Omni 执行器仅实现 dynamic_quantization，显式选择 gpu_ops、ane、streaming 或联合配置时拒绝。重启目标服务才消费新文件，不热改正在运行的模型。

共享 `quantization.runtime.load_profile(path, scope=实际执行scope)` 统一检查 owned0600、4MiB、非符号链接及重复 JSON 键，支持普通 schema2 或 `{"profiles": {完整scope: profile}}` 集合。源 FP16 不转 BF16；搜索分别实测 W4/W8，未通过固定1%双门槛或没有实际存储收益时保留 FP16 identity。允许全部保留 FP16，不称量化收益。生产须同时具备匹配 DynamicEngine 源码/完整权重/配置 scope 和本模型整图批准；局部搜索产物 `approved=False` 不能消费。旧 W8 批准不能继承，源码变化须重新校准。

### 准备和完整校准命令

以下命令串行执行；会构建私有 wheel / 加载真实 7B，但不启停服务、不下载模型、不调用共享 prepare/update。须使用来源兼容的共享基础环境；当前固定来源不匹配，须先解决上述阻塞。

```bash
# 从项目根目录执行；只构建本模型拥有的私有 overlay。
base="$VLLM_BACKEND_PYTHON"
runtime="$(VLLM_BACKEND_PYTHON="$base" "$base" -I composes/seedvr2/prepare_runtime.py)"

# 权重必须预先完整存在，且 model_index.json 已经由本目录 prepare.py 正确建立。
# 输入必须真实存在；图片和5帧CFR视频原始宽高均须小于指定输出宽高。
# 这里624x624可容纳512x512图片的真实超分，仍在现有单帧407040像素门禁内。
VLLM_BACKEND_PYTHON="$base" "$runtime/bin/python" -I extensions/vllm_omni/seedvr2/prepare_quantization.py \
  --model "$HOME/.cache/modelscope/hub/models/Comfy-Org/SeedVR2" \
  --image /实际路径/真实图片.png \
  --video /实际路径/真实5帧CFR视频.mp4 \
  --width 624 --height 624 \
  --output outputs/seedvr2-dynamic-独立新目录

# 仅在已授权启动本模型时执行；默认off，不自动启用。
VLLM_BACKEND_PYTHON="$base" composes/seedvr2/start.sh
# 明确启用时设置 METAL_MODEL_OPTIMIZERS=dynamic_quantization METAL_QUANTIZATION=dynamic METAL_QUANTIZATION_PROFILE=/绝对路径/批准.json
```

离线工具在任何模型加载前核对固定发布完整 SHA256 和索引；不将3B缓存当7B，不自动生成、裁剪或缩小输入。真实图片一帧和最小合法5帧CFR视频，各跑 seed 7723/7724，输出宽高均须大于输入。每候选从四次真实完整生成各捕获首个原 dtype、完整 shape FP16 激活，step 0/1/2/3，使用独立 owned0700 `capture-N` 目录和非 pickle owned0600 文件有界落盘：单个完整 CPU 激活仍最多128MiB，总落盘最多1GiB，不抽样、不转BF16、不提高内存上限。DynamicEngine 一次读取一个完整激活，逐步实测 W4/W8 并允许保留FP16，不建立全样本 CPU/GPU 输入或参考输出列表。成功或失败均移除捕获 hook、关闭并清理本候选临时存储；`capture-N.json` 保留严格JSON的shape、dtype、step及完整性指纹元数据，不保留激活。捕获不得改变完整原精度输出。选中精度随后在原模型每次实际投影 hook 上重新比较完整输出，保留两输入/两seed的全部实际calls和局部双1%门槛；原始基线正反序同seed须逐位一致。

每个局部合格候选都运行完整 VAE encode → DiT → VAE decode → 原生颜色校正/RGB 输出与完整重复生成；最终 relative L2 与按参考 peak 归一 max-abs 均不超过1%，各样本实际投影 calls >0。有限候选中按完整输出最坏误差选一个（不强制所有候选量化）。量化路径检查所选原浮点 module/weight 弱引用均消失；FP16 identity 明确保留原模块。反序完整复验须与候选输出逐位一致，再从原 safetensors 重读所选投影、保持选中精度和引擎审计重建最终摘要。父监督器确认零退出后才同目录原子 no-replace 发布 owned0600 批准文件；已有路径/符号链接拒绝、不覆盖。失败只保留证据，不发布或自动启用。

profile 绑定两份完整真实权重路径/发布哈希、索引、私有模型/校准器源码、执行策略、通用量化实现、Torch/MLX/后端安装路径和版本及设备环境；记录有损，不称无损或全模型加速。证据目录保存 `environment.json`、逐候选误差、每个完整图片/视频 NPY 与逐帧 PNG、`result.json`、`worker.log`、`stage.json`、`supervisor.json`；视频保留五帧及源 fps。校准原生 pipeline 不等于 HTTP 验收，仍需本 README 两个 HTTP 请求通过；没有测得速度或全模型峰值改善。

离线启动要求 **available RAM ≥ 权重总字节 +10GiB**（约25.82GiB），Metal 推荐预算 ≥ 权重+4GiB；运行期间 available ≥4GiB、被监督校准进程树 RSS ≤权重+10GiB、磁盘剩余 ≥2GiB。总校准最长7200秒，单阶段最长1800秒。仅终止自己创建的校准子进程，不降低门槛、不改变水位、不向其它服务发信号。这些保守预算不是实测峰值保证；量化发生在完整 FP16 加载后，不能宣称流式低峰值加载。

### ANE 与联合模式

本 Omni 执行器尚未实现公共 ANE profile 合同，联合生产配置明确拒绝；旧联合校准和批准文件不能当作当前生产入口。`joint.py`、`joint_calibration.py` 与完整校准器随私有 Omni wheel 打包，ASSETS/source SHA、overlay fingerprint、ready 校验和 wheel RECORD 同步覆盖；私有 worker shutdown 显式关闭模型桥。
