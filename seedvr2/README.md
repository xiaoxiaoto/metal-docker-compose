# SeedVR2 7B FP16 · Omni + Metal

本目录提供 Apple Silicon 单 MPS 的原生 `SeedVR2Pipeline` 部署代码，规范模型 ID 为 `seedvr2`，提供图片超分与短视频修复入口。**本次仅编写代码，未下载真实权重、未安装/激活运行时、未启动服务，也未完成真实权重、图片超分或视频验收。** 健康检查成功不能替代推理验收。
根目录将本模型配置为 **dynamic，空闲300秒卸载**，详见[通用生命周期](../README.md#通用模型生命周期)。根目录 `./start.sh seedvr2` 在缺少 plist 时自动调用本目录 `start.sh --prepare-only` 准备软件、私有 runtime、权重和 plist；已有 plist 则校验复用，不隐式修复、更新或加载模型。首次实际推理才离线加载，已有外部服务不被接管，FP16 和量化策略不变。准备成功不代表真实推理已验收。

冷状态的聚合列表不含本模型，单模型查询返回503且不唤醒；先查看根目录 `python3 -I lifecycle_control.py status`，发送下文真实图片/视频curl，完成后复查身份。冷请求在热推理预算之外另留最多1800秒加载预算；不旁路8100推理。动态启动额外要求至少24GiB保守空闲内存，不提高MPS水位。


## 固定模型与安全边界

- 来源 [Comfy-Org/SeedVR2](https://modelscope.cn/models/Comfy-Org/SeedVR2)，固定 revision `5360e873f15685f17acdb077b5e46d16eb6d05d7`；仅普通 7B FP16 checkpoint，不支持 sharp、预量化 checkpoint 或 3B。独立 DynamicEngine 仅消费真实批准 profile，不解除原生 config 量化拒绝。原始模型卡为 [ByteDance-Seed/SeedVR2-7B](https://huggingface.co/ByteDance-Seed/SeedVR2-7B)。
- 只下载 `diffusion_models/seedvr2_7b_fp16.safetensors`（16480583960 字节，SHA256 `2742ca6fee63bc5cc1773f426dd4b07b78cad27f51c9ea5cd42b035e6b592252`）和 `vae/ema_vae_fp16.safetensors`（501324814 字节，SHA256 `20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1`）。下载发生在用户实际执行启动/准备时，不是本次编写时。
- 保留作者缓存路径 `~/.cache/modelscope/hub/models/Comfy-Org/SeedVR2`。完整缓存流式复核 SHA256 后复用；大小正确但校验失败的文件直接报错，不静默覆盖。下载前要求完整权重加 5 GiB 可用磁盘，不自动删除缓存。
- 不执行缓存 Python、不反序列化 pickle、不加载真实张量。检查路径边界、safetensors header、dtype、shape、offset、精确文件大小和固定发布哈希。`model_index.json` 原子创建为 `{"_class_name":"SeedVR2Pipeline","seedvr2_variant":"7b-comfy-fp16"}`；既有内容不同则拒绝，不覆盖用户配置。
- 模型专属适配仅在私有 Omni overlay wheel；不修改共享已安装包、不覆盖 registry、不修改相邻 `vllm-omni-metal`，没有独立 HTTP adapter。
- 适配核对官方 SeedVR 源码 commit `e4de8c24441a67e1b7df56abea10645059bb1185`（`configs_7b/main.yaml`），Omni 基线为实际 0.32 源码 `9534efe802ef79ce52af31847fb144f63d259e90`，私有 wheel 版本为 `0.32.0.dev0+cpu.seedvr2mps2`。7B 保留末层 text 分支，使用带 bias 的 GELU(tanh) MLP、仅视频 pixel RoPE，没有输出 RMSNorm/OutAda；不能把未适配的 3B base 当成 7B pipeline。
- 参考 [Omni SeedVR2 文档](https://github.com/vllm-project/vllm-omni/blob/9534efe802ef79ce52af31847fb144f63d259e90/docs/models/seedvr2.md)和[官方 7B 配置](https://github.com/ByteDance-Seed/SeedVR/blob/e4de8c24441a67e1b7df56abea10645059bb1185/configs_7b/main.yaml)。Omni 基线仅有 3B，本目录私有补丁扩展 7B；基础 Omni commit 不匹配时明确拒绝构建，须重新适配，不能直接用未打补丁的 base CLI 启动 7B。

## 环境、启动和停止

需要 Apple Silicon macOS 15+、MPS、Python 3.9+（启动准备）、ModelScope CLI，以及根目录共享后端/总代理要求的工具。实际执行以下根入口命令会在缺少 plist 时准备软件和权重，模型仍等待首次推理才加载；仅阅读代码无需执行。

```bash
# 首次更新代理路由：会中断在途请求，先确认可重载
./start.sh gateway
./start.sh seedvr2
# 只停止本模型；不停止总代理或其他模型
./stop.sh seedvr2
```

可单独执行 `./seedvr2/start.sh --prepare-only`，并可组合原有 `--quantization off|dynamic` 与 `--quantization-profile 路径` 参数；默认量化策略及环境变量优先级不变。准备模式复用完整流程，在原子写入 plist 并通过 `plutil -lint` 后退出，不 bootstrap 或停止已有 worker；已注册任务和端口冲突仍拒绝，非法参数非零退出。无参数直接运行本目录 `start.sh` 仍准备并加载模型，无需先加载再停止来准备动态部署。

`start.sh` 先校验共享 `VLLM_BACKEND_PYTHON` 和资源，再准备私有 runtime 与权重；CLI 使用私有 `overlay/bin/vllm-omni`，但 `VLLM_BACKEND_PYTHON` 仍指向共享基础环境。launchctl label 为 `com.local.seedvr2-metal`，监听 `127.0.0.1:8100`；plist 位于 `~/Library/Application Support/seedvr2/`，日志为 `~/Library/Logs/seedvr2-metal.log` 和 `seedvr2-metal.error.log`。不是登录自启项。

共享后端预检的既有升级流程不变：版本或源码变化可能停止其他受影响模型，应先确认无在途请求。`--prepare-only` 不通过先启动再停止本模型实现准备，也不接管当前所选服务；共享升级的影响与本模型 worker 的准备/加载分离。

0.32 私有部署的来源指纹包含固定官方 commit、私有版本、补丁、构建器、模型资源及当前后端源码；因此不会复用旧 0.31 overlay 的 `.ready`。复用或新建均检查私有包路径、版本、资源 SHA256 和 wheel RECORD，并运行 `pip check` 后才接受 ready。2026-10-23 已从干净官方源构建并验证修复版 `d330c15663ac65d3`，未修改共享冻结环境或激活服务。图片原精度生成通过不等于 ANE＋DynamicEngine 联合批准；质量 1% 和整模型收益 3% 门槛不变，量化/ANE 默认仍 off。

准备及显式加载的设备能力预检保守要求：权重总字节数加 **10 GiB RAM**，以及权重加 **4 GiB Metal 推荐预算**；`--prepare-only` 不跳过这些检查，下载磁盘检查也保留。这只是预检门槛，**不是测得峰值或可运行保证**。权重和矩阵投影固定 float16，7B 文本分支门控输出与残差保留 float32，归一化后在进入投影前转回 float16；不是全模型升精度。保持 eager、单 GPU、uni executor、TORCH_SDPA 和 VAE tiling；不启用 slicing、offload、原生 config 量化或 MPS fallback，不调整内存水位。动态量化默认 off，仅显式 dynamic 消费批准文件。启动后离线加载、spawn worker，最多等待 1800 秒；失败仅清理本次成功注册的任务。已有注册任务或占用端口导致明确失败，不杀进程。

### 7B 图片数值修复与本次验证

旧 FP16 执行在第3个 DiT block 的文本 MLP Ada-out 门控乘法首次产生非有限值，
之后最终 uint8 转换掩盖异常，完整图片成为单色。实际门控输入仍有限，
范围为 `[-20544, 5504]`。私有补丁只对 NormalMLP 的 FP16 文本路径保留
FP32 门控输出和残差，直到下一次归一化；矩阵输入和全部模型参数仍为 FP16，
视频分支、BF16 路径及原3B SwiGLU 路径不因此改动。不采用 clamp、nan_to_num、
CPU fallback 或整模型升精度，也不宣称与官方 CUDA BF16 autocast 逐位等价。

完整真实猫图验证：输入为用户提供的512×512 Qwen生成图，原图未裁剪或缩小，
输出624×624 RGB。修复私有wheel在无诊断hook、无实例临时覆写条件下，
按seed7723→7724→7724→7723完成四次原生 VAE encode→DiT→VAE decode生成；
同seed两次输出SHA256逐位一致，单次耗时17.81–18.31秒，仅为本次观测，
不构成加速对照。监督零退出、运行最低可用内存约7.85GiB，分布式环境正常关闭。
证据在相邻后端 `outputs/seedvr2-image-fixed-20261023/`；图片与NPY不随包分发。

新增数值回归覆盖超过FP16范围的文本残差及下一block归一化/投影边界，
旧私有wheel重现Inf失败、修复wheel通过；SeedVR2消费者与真实MPS小投影回归
共32项通过。正常准备器已验证补丁应用、非editable安装、完整RECORD/资源/许可证
及依赖检查。上述图片数值修复阶段未验证视频、HTTP、ANE或DynamicEngine联合执行，未发布批准profile；后续HTTP验证见下节。
已有批准的scope包含全部模型Python源码，精度边界改变后必须重新校准，不能复用旧批准。


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

### 2026-10-23：原生HTTP图片与派生视频实测

共享扩散runner已修复强制CPU generator导致MPS采样失败的问题；原生seed-only
请求使用MPS RNG，显式CPU噪声仍受支持，不跳过启动warmup。修复版私有环境为
`b923e9a3bd43ea71`。上述curl使用`--form-string 'prompt= '`，保留必需的单空格；
普通`--form`会去除该字段尾随空白，实际返回缺少prompt的HTTP400。

本次通过私有`vllm-omni serve --omni`在空闲18100端口测试，未改现有部署。
图片输入仍为用户提供的512×512猫图；视频经用户明确授权，从
`~/Downloads/Mask.mp4`截取前5帧并缩至512×512，保持24fps、原文件不变。
输出均为624×624，seed7723各重复两次：

| 原生HTTP接口 | 两次端到端耗时 | 输出核验 |
| --- | --- | --- |
| `/v1/images/edits` | 18.287 / 18.204秒 | RGB PNG；两次像素SHA256一致 |
| `/v1/videos/sync` | 65.301 / 64.404秒 | 5帧、24fps MP4；两次解码像素SHA256一致 |

监督最低可用内存约5.49GiB，未触及4GiB停止线；API与引擎正常关闭，
测试服务退出码0。证据在相邻后端`outputs/seedvr2-http-full-20261023/`。
这是派生图形短片的实际执行/重复性证明，不代表原1024×1024、140帧视频、
自然视频质量或ANE＋DynamicEngine联合批准；不据单次耗时宣称加速。


## 无副作用检查与验证记录

`./seedvr2/start.sh --help`、`./seedvr2/stop.sh --help`、`python3 seedvr2/prepare.py --help` 无部署副作用。`prepare.py --check-only` 仅校验现有缓存，不下载、不写文件（完整真实缓存哈希读取会较慢）。

小型 safetensors、后端和路由回归在本次编辑期间临时编写并执行，验证完成后已按 `AGENTS.md` 的交付清理规则删除测试脚本，不作为部署依赖保留。以下仅记录本次实际结果，不提供指向已删除测试文件的执行命令；图片与视频 curl 均保留。

本次实际通过：缓存回归 9 项、后端回归 11 项（包含 tiny FP16 NaDiT 的 CPU/MPS 对照、3B 保持、图片/视频输出合同）、路由边界 13 项；补丁在临时副本通过 `git apply --check` 并应用，7B DiT 1128 项与 VAE 250 项的完整键/shape 对上固定发布的 header 元数据。只读取了远程 header，没有下载权重张量。

Shell/JS/Python 语法、两份 Compose 配置、无副作用帮助入口通过；命令替身冒烟覆盖总启动模型缺失/格式错误、反向停止及失败后继续清理。合成像素的模型后处理经原生图像提取和 PNG 编码后可解码，**不是实际超分推理**。未构建/安装私有 wheel、未启动 Nginx/模型，也未验证真实 HTTP、完整 7B、真实图片/视频质量或峰值内存。共享依赖导入仍有 PyAV/OpenCV/Homebrew FFmpeg 的 Objective-C 重复类警告，本次未改依赖规避。

本次代码交付不引用其他模型或历史测试作为 SeedVR2 的真实权重验收证据。使用前另行核对模型许可证与数据使用约束。

## DynamicEngine：真实批准后才开启

新增 `seedvr2mps2` 私有 wheel 将本目录 `quantization.py`、安全权重清单和完整图校准器一同打包；准备器来源指纹和 `.ready` 检查绑定各文件 SHA256。只在实际 `load_weights` 完成后替换选中投影，不修改已安装上游，不注册第二套模型；请求阶段不读 profile、不校准、不编译、不构造权重。
私有源码固定为实际0.32官方提交 `9534efe802ef79ce52af31847fb144f63d259e90`；准备器从相邻后端 checkout 的干净副本构建非 editable 后端 wheel，只安装到本模型私有 overlay。指纹覆盖 `pyproject.toml`、完整 Python/native 资源、README 和许可证；ready 校验私有导入路径、随包文件哈希、许可证及安装 RECORD。共享 base/current 不被修改；`.pth` 只复用基础依赖，不冒充完整依赖重装。
安装/ready校验使用选定Python `-I` 并导入通用 `quantization.capture`、
`adapter`、`runtime`。源FP16/BF16保持dtype，通用FP32只identity；
这些安装检查不替代本模型真实7B整图质量与速度门禁，不自动启用或发布。
动态profile路径必须为绝对路径；相对路径明确拒绝。

候选仅为 36 个 block 中第 0/12/24/35 层的 `mlp.vid.proj_out`，最多批准其中一个。7B 保持带 bias 的 `Linear → GELU(approximate="tanh") → Linear`，没有 SwiGLU/gate-up 假设；text、QKV、VAE、其它线性层不量化。普通投影返回 Tensor，拒绝 vLLM tuple/skip-bias Linear 子类，不伪造 tuple 适配。应用前检查完整模块树的 module、parameter、buffer alias；全部校验和量化构造成功才提交，失败保留原图。

`SEEDVR2_METAL_QUANTIZATION=off|dynamic`（启动器 `--quantization`），默认 `off`，不读取 profile；`dynamic` 必须明确提供已批准 profile。`SEEDVR2_METAL_QUANTIZATION_PROFILE` / `--quantization-profile` 指定路径；私有参数未设置时分别回退 `VLLM_OMNI_METAL_QUANTIZATION` / `VLLM_OMNI_METAL_QUANTIZATION_PROFILE`。模型显式 off 覆盖全局 dynamic。旧 W8-only 环境变量和启动 flags 已移除。重启目标服务才消费新文件，不热改正在运行的模型。

共享 `quantization.runtime.load_profile(path, scope=实际执行scope)` 统一检查 owned0600、4MiB、非符号链接及重复 JSON 键，支持普通 schema2 或 `{"profiles": {完整scope: profile}}` 集合。源 FP16 不转 BF16；搜索分别实测 W4/W8，未通过固定1%双门槛或没有实际存储收益时保留 FP16 identity。允许全部保留 FP16，不称量化收益。生产须同时具备匹配 DynamicEngine 源码/完整权重/配置 scope 和本模型整图批准；局部搜索产物 `approved=False` 不能消费。旧 W8 批准不能继承，源码变化须重新校准。

### 准备和完整校准命令

以下命令由操作者串行执行；会构建私有 wheel / 加载真实 7B，但**不启停服务、不下载模型、不调用共享 prepare/update、不停止 YuE2 或其它服务**。本次选定已有基础环境为 `runtime/0e63615c03ef50e1`，准备器仍严格核对官方 Omni 固定 commit；不可用或不兼容时明确失败，不自动切换共享环境。

```bash
# 从 docker-compose 根目录执行；只构建本模型拥有的私有 overlay。
base="$HOME/.local/share/local-model-backend/runtime/0e63615c03ef50e1/bin/python"
runtime="$(VLLM_BACKEND_PYTHON="$base" "$base" -I seedvr2/prepare_runtime.py)"

# 权重必须预先完整存在，且 model_index.json 已经由本目录 prepare.py 正确建立。
# 输入必须真实存在；图片和5帧CFR视频原始宽高均须小于指定输出宽高。
# 这里624x624可容纳512x512图片的真实超分，仍在现有单帧407040像素门禁内。
VLLM_BACKEND_PYTHON="$base" "$runtime/bin/python" -I seedvr2/prepare_quantization.py \
  --model "$HOME/.cache/modelscope/hub/models/Comfy-Org/SeedVR2" \
  --image /实际路径/真实图片.png \
  --video /实际路径/真实5帧CFR视频.mp4 \
  --width 624 --height 624 \
  --output outputs/seedvr2-dynamic-独立新目录

# 仅在已授权启动本模型时执行；默认off，不自动启用。
VLLM_BACKEND_PYTHON="$base" ./seedvr2/start.sh --quantization off
# 明确启用时另传 --quantization dynamic --quantization-profile /实际批准路径
```

离线工具在任何模型加载前核对固定发布完整 SHA256 和索引；不将3B缓存当7B，不自动生成、裁剪或缩小输入。真实图片一帧和最小合法5帧CFR视频，各跑 seed 7723/7724，输出宽高均须大于输入。每候选从四次真实完整生成各捕获首个原 dtype、完整 shape FP16 激活，step 0/1/2/3，使用独立 owned0700 `capture-N` 目录和非 pickle owned0600 文件有界落盘：单个完整 CPU 激活仍最多128MiB，总落盘最多1GiB，不抽样、不转BF16、不提高内存上限。DynamicEngine 一次读取一个完整激活，逐步实测 W4/W8 并允许保留FP16，不建立全样本 CPU/GPU 输入或参考输出列表。成功或失败均移除捕获 hook、关闭并清理本候选临时存储；`capture-N.json` 保留严格JSON的shape、dtype、step及完整性指纹元数据，不保留激活。捕获不得改变完整原精度输出。选中精度随后在原模型每次实际投影 hook 上重新比较完整输出，保留两输入/两seed的全部实际calls和局部双1%门槛；原始基线正反序同seed须逐位一致。

每个局部合格候选都运行完整 VAE encode → DiT → VAE decode → 原生颜色校正/RGB 输出与完整重复生成；最终 relative L2 与按参考 peak 归一 max-abs 均不超过1%，各样本实际投影 calls >0。有限候选中按完整输出最坏误差选一个（不强制所有候选量化）。量化路径检查所选原浮点 module/weight 弱引用均消失；FP16 identity 明确保留原模块。反序完整复验须与候选输出逐位一致，再从原 safetensors 重读所选投影、保持选中精度和引擎审计重建最终摘要。父监督器确认零退出后才同目录原子 no-replace 发布 owned0600 批准文件；已有路径/符号链接拒绝、不覆盖。失败只保留证据，不发布或自动启用。

profile 绑定两份完整真实权重路径/发布哈希、索引、私有模型/校准器源码、执行策略、通用量化实现、Torch/MLX/后端安装路径和版本及设备环境；记录有损，不称无损或全模型加速。证据目录保存 `environment.json`、逐候选误差、每个完整图片/视频 NPY 与逐帧 PNG、`result.json`、`worker.log`、`stage.json`、`supervisor.json`；视频保留五帧及源 fps。校准原生 pipeline 不等于 HTTP 验收，仍需本 README 两个 HTTP 请求通过；没有测得速度或全模型峰值改善。

离线启动要求 **available RAM ≥ 权重总字节 +10GiB**（约25.82GiB），Metal 推荐预算 ≥ 权重+4GiB；运行期间 available ≥4GiB、被监督校准进程树 RSS ≤权重+10GiB、磁盘剩余 ≥2GiB。总校准最长7200秒，单阶段最长1800秒。仅终止自己创建的校准子进程，不降低门槛、不改变水位、不向其它服务发信号。这些保守预算不是实测峰值保证；量化发生在完整 FP16 加载后，不能宣称流式低峰值加载。

2026-10-23：私有 runtime `b923e9a3bd43ea71` 的真实 HTTP 原图及用户授权派生五帧视频生成、同seed复验已通过，证据为相邻后端 `outputs/seedvr2-http-full-20261023/`。随后完整 joint 校准在 `outputs/seedvr2-joint-full-20261023/stage.json` 的 `candidate-0-capture-video-7723` 因旧版四样本合计128MiB CPU捕获门禁拒绝，尚未开始search/ANE，不是联合质量或速度通过。

2026-10-24：有界落盘和流式搜索已安装至独立 runtime `32d4415b8e04479a`，正常准备器ready复用及ANE工具准备通过。因GitHub传输失败，Omni由已核验整包SHA256、完整RECORD、同一固定commit与7B补丁的缓存纯Pythonwheel更新当前校准资源后重新打包，**不是全新Omni源码构建**；后端wheel仍从干净当前源码构建。离线来源记录见相邻后端 `outputs/seedvr2-stream-install-20261023/offline-wheel-source.json`。

本轮完整真实校准证据为 `outputs/seedvr2-joint-streamed-20261024/`：四个原精度workload均已输出，停在 `baseline-repeat-video-7724`，运行约293秒后available降至2564866048字节（2.39GiB），低于4GiB停止线；监督器终止本次子进程，退出码-15。**尚未进入捕获、search或ANE联合执行，未发布批准或自动启用服务**。完整模型上的流式路径、双1%质量、相同量化GPU对照稳定快3%及联合生命周期仍未验收；不会降低输入、四workload或资源门槛。后端递归来源指纹、ready/RECORD及校准器ASSETS绑定覆盖新API与资源，旧实现批准不可迁移。

### 2026-10-22历史可达性记录（当前已补齐7B）

以下缺权重结果仅为当时记录。2026-10-23固定7B与VAE已完整下载并通过校验，
不能再把下列历史拒绝当作当前缺权重证据；实际图片/HTTP验证见上文。

本次定向检查：默认 ModelScope `Comfy-Org/SeedVR2`、SeedVR2 私有 overlay、`~/Library/Application Support/seedvr2` 和对应 LaunchAgent 均未发现部署；没有提供自定义 SeedVR2 路径。已有 HF `models--numz--SeedVR2_comfyUI/snapshots/09ced71023636e9bc8cdf9cdecfb2625d1e691e8` 仅列出 `seedvr2_ema_3b_fp16.safetensors` 与 VAE，不能替代缺失的固定 7B DiT。还定向检查了 `~/Models` 的 SeedVR2 名称、HF/ModelScope SeedVR2 命名缓存；未全盘扫描或读取私人照片库。

历史缓存观察不作为当前校准证据。本次迁移未运行准备器、构建私有环境、加载7B或HTTP，不创建批准profile，**不能宣称SeedVR2已开启动态量化或质量通过**。持久边界测试位于相邻后端 `tests/test_seedvr2_quantization.py`；CPU测试仅检验消费者/事务，显式MPS小图测试也不替代完整7B门禁。

只读核对当前缺权重门禁可直接使用固定基础 Python（不需要真实图片/视频存在，也不构建 overlay）：

```bash
base="$HOME/.local/share/local-model-backend/runtime/7cbda79691696d01/bin/python"
"$base" -I seedvr2/prepare_quantization.py \
  --model "$HOME/.cache/modelscope/hub/models/Comfy-Org/SeedVR2" \
  --image /nonexistent/seedvr2-input.png \
  --video /nonexistent/seedvr2-five-frames.mp4 \
  --output "/tmp/seedvr2-weight-preflight-$$"
```

选择一个尚不存在的 `--output` 路径。当前应先因缺少固定7B索引/权重而非输入视频拒绝，在创建证据目录、导入模型或启动子进程之前退出；此拒绝不是校准或启用成功。

### 显式 ANE + DynamicEngine 联合路径（默认仍 off）

`--quantization joint --quantization-profile /绝对路径/joint-approved.json`
只消费独立的联合批准，不把量化批准与 ANE 缓存的存在当成联合批准。
ANE 执行未量化原 FP16 **带 bias 的 proj_in**，GPU 保留原 `GELU(tanh)`，
DynamicEngine 执行实际 W4/W8 `proj_out`；identity down 不得批准联合执行。
每次生产投影必须有真实非零 ANE 命中，未批准上下文、形状变化、缺桥或零分摊明确拒绝，
不静默回落后声称已开启。请求边界使用真实 `prepared_layout.frame_count`。

离线命令在现有 `prepare_quantization.py` 参数上增加 `--joint`，
为真实图片/五帧视频 × seed 7723/7724 运行完整校准。scope 在任何量化替换前
绑定原模型发布权重、执行配置、两实现和校准器源码。完整 GPU-only split / joint
各独立 warmup 后执行 ABBA，计时来自 MPS 同步后的实际 pipeline 到 CPU RGB，
不含保存 PNG/NPY。每个工作量必须满足双 1% 完整输出误差、确定性、实际两路命中
和固定至少 3% 全流水线收益；释放浮点 down 后复验，并关闭桥和分布式环境。
父监督器成功后才发布最终联合批准。未跑真实校准，不建立或默认启用任何批准。

`joint.py`、`joint_calibration.py` 与完整校准器均由私有 Omni wheel 正常打包；
ASSETS/source SHA、overlay fingerprint、ready 校验和 wheel RECORD 同步覆盖，
不在已安装包复制源码。私有 worker shutdown 显式关闭模型桥；垃圾回收只作兜底。

### 2026-10-22 本次安装与权重前置拒绝

正常准备器已完成0.32私有 `overlays/seedvr2/1079933f5a52529a` 非 editable wheel
构建、安装、来源/完整RECORD/许可证及依赖校验，联合资源随包齐全；没有修改基础环境。
实际执行该环境的 `prepare_quantization.py --joint`，在创建证据目录、模型worker或
读取输入之前，因缺少原始7B `model_index.json` 拒绝。当前默认ModelScope目录不存在，
定向HF缓存仅有3B DiT和VAE，不能冒充7B；未提供真实原图和五帧CFR视频。
没有自动下载权重、生成/缩小素材、加载7B、发布profile或启用默认。
本次输出在相邻后端 `outputs/joint-ane-dynamic-20261022-222638/seed-real-preflight.log`，
安装输出为同目录 `seedvr2-runtime-install.log`。上文“未构建私有wheel”属于历史记录，
不能据此否认本次安装，也不能把本次安装当作完整真实超分通过。

