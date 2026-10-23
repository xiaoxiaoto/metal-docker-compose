# Qwen-Image-2.1-Turbo · vLLM-Omni + Omni Metal

Apple Silicon 上运行上游原生 **QwenImage21Pipeline**，复用共享隔离后端的 **vLLM-Omni + vllm-omni-metal**。不使用普通 Qwen-Image pipeline、CUDA Docker、自建 HTTP adapter、CPU fallback 或 offload。官方 checkpoint 保持原精度；动态混合精度默认 off，仅消费本机真实整图校准批准的 DynamicEngine profile。

- 官方在线示例：https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/text_to_image
- 官方模型卡：https://modelscope.cn/models/Qwen/Qwen-Image-2.1-Turbo
- 核对示例时的 Omni main：`1b6c6c50cf7aaaff82d579e41072a2fe0057d212`；实际运行版本由根目录联网预检和 `check_versions.py` 校验，不强行固定到未经共享后端验证的组合。
- 对外 ID：**`qwen-image-2.1-turbo`**。下载仓库为 `Qwen/Qwen-Image-2.1-Turbo`，不把普通 Qwen-Image 示例中的模型名或 50 步参数照搬到本模型。
- 单 generation pipeline，从 `model_index.json` 解析组件与采样；无需双阶段 Omni profile，不增加无需求的 `metal-deploy.yaml`。

根目录将本模型配置为 **dynamic，空闲300秒卸载**，详见[通用生命周期](../README.md#通用模型生命周期)。根目录 `./start.sh qwenimage` 在缺少 plist 时自动调用本目录 `start.sh --prepare-only` 准备软件、私有 runtime、权重和 plist；已有 plist 则校验复用，不隐式修复、更新或加载模型。首次推理才离线加载，不改变已有精度和量化策略，外部已运行服务不被接管。

冷模型不在聚合列表，单模型身份查询返回503且不唤醒；先查看根目录 `python3 -I lifecycle_control.py status`，发送下文真实文生图curl，完成后复查身份。冷请求另留最多600秒加载预算；不要旁路8099推理。动态启动额外要求至少36GiB保守空闲内存，不能为了常驻H2O而提高MPS水位或放宽门槛。

## 环境与资源

Apple Silicon、macOS 15+、Python 3.9+（模型准备脚本）、ModelScope CLI。网关另需 Docker Desktop。复用共享非 editable 基础环境的依赖与官方 Omni commit；`prepare_runtime.py` 在 Qwen 拥有的私有 overlay 内分别安装当前相邻后端和打补丁 Omni 的非 editable wheel。基础环境缺 diffusion／DynamicEngine API 时也不调用根准备器升级 current；兼容组为 vLLM/Metal `0.32.0`、Omni/Omni Metal `0.32.0.dev0`，私有 Omni 使用 `0.32.0.dev0+cpu.qwenmetal1`。不修改共享安装或其他模型环境。

模型卡的 Diffusers 直接运行示例要求开发版 Diffusers 与 Transformers ≥5.17；本部署走 **Omni 的原生实现**，不是 `diffusion_load_format=diffusers`，不能为了照搬该示例逐包升级共享依赖。原生组件能导入、小算子通过都不等于真实权重能完成生成。

本次 ModelScope 文件清单的权重总计 **32,440,133,648 字节（30.21 GiB）**：Qwen3-VL encoder、两片 transformer 与 RGBA VAE。48 GiB Mac 的 Metal 推荐预算约 37.44 GiB，不是最低内存或生成质量承诺。首次先测试 512×512、单图，不与其他大模型并驻；官方 2K 预设仍待独立资源/质量验收。

准备及显式加载均检查物理内存至少为权重预算 +10 GiB、Metal 推荐预算至少为权重预算 +4 GiB；这些是设备能力预检，不代表测得的峰值上限，`--prepare-only` 也不跳过。下载/补全前的剩余磁盘至少完整权重 +5 GiB 检查仍保留（不完整缓存也按完整下载预算检查），不自动清理缓存。

`prepare.py` 固定选择 `~/.cache/modelscope/hub/models/Qwen/Qwen-Image-2.1-Turbo/`，完整缓存直接复用；否则下载/补全官方 `master`，排除 `assets/*` 示例媒体。静态校验组件架构、未量化配置、8 步递减有限 `sample_sigmas`、processor/tokenizer、权重索引、路径与 safetensors 数据边界，不执行缓存代码或反序列化 pickle。不自动换仓库或选择其他版本；`master` 非不可变 revision，首次下载需核对仓库与研究许可。

原生模型保留 Qwen3-VL encoder、transformer、RGBA VAE、scheduler 和官方 `sample_sigmas`，不手写采样网格。配置为 bf16、单设备 `uni`、`TORCH_SDPA`、eager、VAE slicing/tiling；不删除 unsupported 校验，不接受上游量化 checkpoint/config，不启用 offload、并行或提高 MPS 内存水位。下述独立动态接入只在官方未量化权重加载完成后应用已批准的 down 混合精度组合。

完整联合校准采用按候选层串行捕获：每层各执行一次完整512×512、
官方8步原精度生成，保留该层0/2/4/6步的完整真实激活；四层全部搜索结束后
才安装合并的DynamicEngine候选。每批仍限制128MiB激活，不同时保留四层
至少384MiB的捕获，不缩小模型、图片、候选层集合或step覆盖。每次捕获生成
均与原精度基线校验整图哈希。代价是增加完整生成次数；启动32GiB、
运行可用内存4GiB/RSS42GiB、整图双1%及联合收益3%门槛不变。
该资源调整本身不构成完整模型通过证明，批准仍以本次完整监督运行证据为准。

本次完整模型实测证据：
`vllm-omni-metal/outputs/qwen-joint-resource-20261023-114733/`。
私有非editable环境指纹为`a5121d23d8b769c7`，四层均搜索选择W8A16，
每个联合正式样本实际执行32次量化投影及32次ANE调用（32块、8192行）。
运行549.58秒，峰值进程RSS17.32GiB、最低可用8.01GiB；监督未触及资源门槛，
引擎已关闭。这是完整原生模型生成，不是单层或随机权重验证。

| 完整pipeline实测 | 结果 |
| --- | --- |
| 原精度GPU两次 | 29.840 / 30.032秒 |
| 同量化GPU两次 | 30.148 / 29.984秒 |
| ANE＋DynamicEngine两次 | 30.281 / 29.557秒 |
| 联合整图相对L2（两次相同） | 0.62369%，通过1% |
| 联合整图scaled-max（两次相同） | 10.58824%，未通过1% |
| 相对同量化GPU的保守稳定收益 | -0.98879%，未达到3% |

两次联合输出哈希一致，但局部投影搜索通过不等于整图误差通过。
本次联合校准拒绝，未发布批准profile、未激活部署；因收益门槛先拒绝，
未执行批准后的浮点基线释放与释放后复现阶段。不得将完成完整模型运行
写成联合方案已通过或已获得整模型加速。

## 启停

从项目根目录执行：

```bash
./start.sh gateway  # 首次更新路由需构建网关；若网关已运行，会中断在途请求，先确认
./start.sh qwenimage
./stop.sh qwenimage # 只停止本模型，不停止网关或其他模型
```

可单独执行 `./qwen-image-2.1-turbo/start.sh --prepare-only`：复用完整准备流程，在原子写入 plist 并通过 `plutil -lint` 后退出，不 bootstrap 或停止已有 worker。已注册任务和端口冲突仍拒绝，非法参数非零退出。无参数直接运行本目录 `start.sh` 仍准备并加载模型；无需先加载再停止来准备动态部署。

先确认大权重下载、磁盘与内存预算，不为验证单模型启动 `all`。共享后端预检发现版本/源码变化时会停止本项目其他模型并构建新环境，失败不激活；只重启所选目标。直接运行本目录 `start.sh` / `stop.sh` 只管理该模型，不操作网关。

服务绑定 `127.0.0.1:8099`，label 为 `com.local.qwen-image-2.1-turbo-metal`。plist 在 `~/Library/Application Support/qwen-image-2.1-turbo/` 原子写入，不是登录自启项。保持 spawn、离线加载、600 秒启动等待、不自动重启失败进程；已注册任务/被占用端口明确报错，不杀其他进程，失败只卸载本次启动任务。

日志：`~/Library/Logs/qwen-image-2.1-turbo-metal.log`、`~/Library/Logs/qwen-image-2.1-turbo-metal.error.log`。统一入口只绑定本机，无鉴权，不暴露公网。

## 文生图

模型已加载时，检查不带参数的聚合列表包含 `qwen-image-2.1-turbo`；dynamic冷状态先检查生命周期登记，第一次推理完成后再执行此查询：

```bash
curl --fail-with-body --show-error --connect-timeout 5 --max-time 10 \
  'http://127.0.0.1/v1/models'
```

curl 沿用根目录同类媒体示例：先查单模型身份，再保存二进制输出；成功后才打开图片。

```bash
curl --fail-with-body --show-error --connect-timeout 5 --max-time 10 \
  --get 'http://127.0.0.1/v1/models' \
  --data-urlencode 'model=qwen-image-2.1-turbo'

curl --fail-with-body --show-error --connect-timeout 5 --max-time 1800 \
  'http://127.0.0.1/v1/images/generations' \
  --header 'Content-Type: application/json' \
  --data-binary '{
    "model": "qwen-image-2.1-turbo",
    "prompt": "一只橘猫坐在窗台上，窗外是蓝天，阳光温暖，细节清晰。",
    "size": "512x512",
    "n": 1,
    "num_inference_steps": 8,
    "true_cfg_scale": 1.0,
    "seed": 42,
    "response_format": "file"
  }' \
  --dump-header /tmp/qwen-image-test.headers \
  --output /tmp/qwen-image-test.png &&
open /tmp/qwen-image-test.png
```

`/v1/images/generations` 用 `prompt`、`size`、`n`，参数放顶层，不使用聊天接口的 `messages` / `extra_body`。采样仍采用模型的 8 个 `sample_sigmas`；不要指定其他网格替换 Turbo 计划。`response_format=file`、`n=1` 返回 PNG 二进制，不送入 JSON 格式化管道；若用默认 `b64_json`，返回 JSON 的 `data[0].b64_json`。多图 file 响应可能是 ZIP，本示例不这样调用。

结果保存在 `/tmp/qwen-image-test.png` 和 `/tmp/qwen-image-test.headers`，复测覆盖文件。成功要求 HTTP 200、`Content-Type: image/png`、PNG 可解码、尺寸匹配、RGB/RGBA 有效非空/非单色，以及内容与提示相符；curl 失败不打开图片，不能把错误 JSON 或旧文件当作成功。RGBA 模型不要强行丢弃 alpha。当前代理请求上限 64 KiB、等待 1800 秒。

## 图片编辑：游艇草图转照片

先从项目根目录执行 `./start.sh qwenimage`，并按上文检查不带参数的 `/v1/models` 聚合列表。下面使用 [官方游艇草图](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/resolve/main/assets/turbo-boat-input.png)及给定 Diffusers 提示词，改走原生 `/v1/images/edits`，保留 2048×2048、8 步、CFG=1、CPU generator 和 seed `20261009`。**2K 编辑尚未真机推理验收**；48 GiB 机器先确认内存余量，首次可将 `size` 改为 `512x512`，不要与其他大模型并驻。

```bash
set -o pipefail
edit_prompt='Create a photorealistic photograph of the motor yacht from the input sketch, sailing on open sea, captured in full side profile with the camera at water level. Build the vessel with the structure shown in the input sketch: a long low hull; one tall mast stepped forward with a small cross-tree platform near its top; a wheelhouse superstructure with an open bridge wing just behind the mast; a midships deckhouse carrying a row of four rectangular windows with a lifeboat stowed on its roof; a rectangular funnel behind that deckhouse; an aft deckhouse with a row of three rectangular windows and a second lifeboat above it; thin railings along every deck edge; the hull side pierced by one small square window at the bow, an upper line of five round portholes on the raised forward section, and a lower line of nine round portholes along the main hull. Paint the hull black, the superstructures and the mast white, and the funnel yellow. Set the yacht broadside on a calm deep-blue sea with a low gentle swell and a thin white wake trailing from the bow, the straight horizon crossing behind the funnel top, under a clear sky with soft scattered clouds; sunlight from the front left lights the white superstructures and leaves a bright reflection path on the water beside the hull with a soft contact shadow along the waterline; the entire vessel fits inside the frame with open water in the foreground and sky above.'

curl --fail-with-body --show-error --connect-timeout 5 --max-time 10 \
  --get 'http://127.0.0.1/v1/models' \
  --data-urlencode 'model=qwen-image-2.1-turbo' &&
image_url="$(curl --head --location --fail-with-body --show-error \
  --connect-timeout 5 --max-time 60 \
  'https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/resolve/main/assets/turbo-boat-input.png' \
  --output /dev/null --write-out '%{url_effective}')" &&
curl --fail-with-body --show-error --connect-timeout 5 --max-time 1800 \
  'http://127.0.0.1/v1/images/edits' \
  --form-string 'model=qwen-image-2.1-turbo' \
  --form-string "url=$image_url" \
  --form-string "prompt=$edit_prompt" \
  --form 'size=2048x2048' \
  --form 'n=1' \
  --form 'num_inference_steps=8' \
  --form 'true_cfg_scale=1.0' \
  --form 'seed=20261009' \
  --form 'generator_device=cpu' \
  --form 'response_format=b64_json' \
  --form 'output_format=png' \
  --dump-header /tmp/qwen-image-edit.headers | \
  python3 -c 'import base64,json,pathlib,sys; r=json.load(sys.stdin); sys.exit(json.dumps(r, ensure_ascii=False)) if "data" not in r else pathlib.Path("/tmp/qwen-image-edit.png").write_bytes(base64.b64decode(r["data"][0]["b64_json"], validate=True))' &&
open /tmp/qwen-image-edit.png
```

此编辑端点当前只支持 `response_format=b64_json`，不是文生图的 `file`；管道解码 `data[0].b64_json`，不将 PNG 当 JSON 展示。设置 `pipefail` 保留 curl 非零状态，HTTP 错误体会显示且失败不打开旧图片。输出为 `/tmp/qwen-image-edit.png` 和 `/tmp/qwen-image-edit.headers`，复测覆盖；成功需 HTTP 200、JSON 包含可解码的图片、PNG 尺寸匹配、RGBA 有效非单色，并对照草图检查船体结构、颜色与场景，不仅检查下载成功。

官方草图为 3,200×1,120 RGBA、1,878,998 字节，超过当前 **64 KiB 整个请求体**上限，所以使用后端支持的 `url` 表单字段，不直接上传。当前后端下载器不跟随 HTTP 重定向，示例先用 HEAD 解析最终图片 URL，再让后端下载；签名 URL 可能过期，重试时重新执行整段，不保存或硬编码。客户端和后端均需能访问该站点；自行替换 URL 时只使用可信、可直接返回图片的 HTTPS 地址。不要手设 multipart Content-Type/boundary。本次已核对运行中 OpenAPI，并用实际后端下载器加载此 RGBA 草图，**未执行 2048×2048 编辑生成**。

`use_kv_cache=True` 是 Diffusers 参数，当前 HTTP 编辑接口不暴露该字段；添加同名表单项不能证明已生效。原生 pipeline 根据 transformer 的 `causal_condition` 自动创建每请求 KV cache，本地官方配置为 `true`，无需额外传参。RGBA 条件图由原生预处理保留/转换，不需要在客户端安装 Diffusers；seed 与 CPU generator 参数不代表与 Diffusers 输出逐字节一致。

## 逐模型动态混合精度

`metal_adapter.py`、`quantization_adapter.py`、`joint_adapter.py` 与准备器资源
`prepare_quantization.py`、`prepare_ane.py`、`prepare.py` 随私有非 editable
Omni wheel 正常打包；源文件 SHA、Omni RECORD、overlay 指纹和 `.ready` 同步校验。
实际离线 CLI 从本部署源码目录运行，不从 wheel 资源启动。
安装/ready校验用选定Python `-I` 导入共享 `quantization.capture`、`adapter`、
`runtime`，递归后端源码指纹覆盖新模块；通用入口接通不是任意Omni模型已移植的证明。

当前私有补丁以官方 Omni 提交
`9534efe802ef79ce52af31847fb144f63d259e90` 为唯一来源基线；
准备器明确拒绝其他共享 Omni 提交，不自动升级依赖或重设现有共享环境。
0.32 适配保留官方新增 NPU attention 导入和 NPU rotary 分支，并在其前加入 MPS rotary，
保留 RMSNorm、FFN 与真实 ANE 单 projection + DynamicEngine down 联合执行路径。
源码补丁兼容检查不代表 wheel 安装、真实整图精度／收益或校准批准已通过；
旧 `ebe2c18271db4fdd12221fbb4def6061e14a9379` 安装烟测不作为此基线的证据。
提交、补丁、准备器及适配器均参与 ready 指纹，旧 overlay 不会被当作新版本复用；
本次源码适配不运行准备器、不切换 ready/current，也不启用默认量化或 ANE。

`QWEN_IMAGE_METAL_QUANTIZATION=off|dynamic|joint` 默认 `off`；
`QWEN_IMAGE_METAL_QUANTIZATION_PROFILE=/absolute/approved.json` 指定批准文件。
两个参数分别 fallback 到 `VLLM_OMNI_METAL_QUANTIZATION` 和
`VLLM_OMNI_METAL_QUANTIZATION_PROFILE`，模型显式 off 优先于全局 dynamic。
dynamic 缺 profile、未批准、错误环境／源码／完整 checkpoint／所选权重均报错，
不使用 auto 或静默回退。旧 W8 专用生产入口已删除。

共享 `quantization.runtime.load_profile` 负责 owned0600、非符号链接、4 MiB、
无重复 key／有限 plain JSON 及 scope 校验；支持普通 schema2 或
`{"profiles": {"完整执行scope SHA": profile}}` 集合。只按完整执行 SHA 取配置，
不是按模型名路由。

仅允许单设备、未融合且无 bias 的官方 `transformer_blocks.N.img_mlp.out`：
零复制 `dense_view` 复用真实未量化 vLLM `RowParallelLinear` 权重，
DynamicEngine 逐项实测 W4、W8，对局部双 1% 门槛通过者选择持久存储更小者，
不通过则保留源 FP16/BF16；FP32 只保留、不冒充量化收益。
按每项 descriptor 的实际 bits 构造 affine/group64 量化投影；
identity 项保持原模块、dtype 和权重。保留 `return_bias` Tensor／tuple 与
`skip_bias_add` 合同，不拆 gate/up、attention 融合，不量化 encoder 或 VAE。
拒绝 TP、shared module、tied weight/buffer 和丢弃 hook。
量化包装不持有原浮点模块或 `.weight`；不修改 checkpoint，
不在请求中搜索、量化或编译。

独立 dynamic 与旧 full-FFN ANE 批准不能拼接。`joint` 使用独立整图批准：
ANE 仅处理未量化 gate，GPU DynamicEngine 处理 down；不读取量化 down 的 `.weight`。
gate 的 Tensor／tuple、skip-bias 合同不变，GPU pointwise 仍用 `auto`。
scope 绑定整个本地模型 safetensors、配置、tokenizer、适配器、通用
linear/engine 源码、上游、原生采样和 GPU 路由；实际消费完整读盘哈希。
旧 schema／旧实现批准不可迁移，必须重新校准。
**本次仅交付源码迁移，没有运行真实 QwenImage 校准／部署，没有本次通过的 profile，
不宣称已启用、无损、加速或节省全模型峰值。GPU AOT 不是 INT8。**


### 真实整图离线批准

使用已验证的固定基础环境，不调用共享 `prepare_backend.py` 或升级 current：

```bash
cd /Users/zhaodehua/Documents/docker-compose/qwen-image-2.1-turbo
base="$HOME/.local/share/local-model-backend/runtime/c076adaf5074c803/bin/python"
runtime="$("$base" -I prepare_runtime.py)"
"$runtime/bin/python" -I prepare_quantization.py \
  --layers 0,10,21,31 --size 512x512 --seed 42 \
  --output /Users/zhaodehua/Documents/vllm-omni-metal/outputs/qwen-dynamic-real
```

已存在证据目录会拒绝覆盖；本机批准文件先在同目录完整写入、fsync，再以原子 no-overwrite
hard-link 提交，并发发布也不覆盖赢家。可用 `--profile` 指定新文件。
不需要 `--with-vpipe`，缺 Metal compiler 不应冒充已启用 AOT。若仅安装／验证 overlay，
准备器会真实构建私有 wheel，不触碰已安装上游；上面的校准另由明确命令执行。
安装与 pip/check_versions 校验均使用所选解释器 `-I`；后端通过该解释器的 `pip wheel`
隔离构建，Omni 则由 uv 的隔离构建环境及显式 `--python` 选择解释器。
相邻 `vllm-omni-metal` 的 `pyproject.toml`、README、LICENSE、全部 Python／native 资源
生成 manifest，参与 overlay 指纹；后端从只含 manifest 文件的干净临时副本构建 wheel，
只安装到本模型 overlay。ready 检查后端非 editable 来源、完整资源集合与逐文件 SHA、
随包 LICENSE、后端及 Omni 全部已哈希 RECORD 项；完整 manifest 保存在 `runtime.json`。
基础依赖仍通过明确 `.pth` 复用，不能把此操作称为全量依赖重装。
attention launcher 仅在明确准备阶段构建；不启停服务，不下载模型，不改变共享 base/current。

严格复用 `prepare_ane.py` GPU A/B 监督器：**启动 available ≥32 GiB**、
启动磁盘 ≥4 GiB；运行 available ≥4 GiB、进程树 RSS ≤42 GiB、磁盘保留 ≥2 GiB，
总时限 3600 秒、每图 1800 秒。不会停止其它模型、下载或降低门禁。48 GiB 机器的
其它服务即使释放后仍可能达不到 32 GiB；此时工具在加载前硬拒绝，不生成批准 profile。
达不到资源门槛时必须停止，不能用小型图片或历史原精度请求代替动态量化验收。

工具先保存无诊断 hook 的原精度完整图，再在一次真实完整 8 步生成中
捕获指定候选的第 0/2/4/6 次完整 CPU 激活，保持源 dtype、形状和原始 bytes，
总捕获上限 128 MiB；捕获 hook 整图必须与原生图 hash 相同。
DynamicEngine 搜索所有指定候选，保留逐位宽局部审计和激活指纹；
不是全层穷尽搜索，局部通过不批准。允许全源精度保留，不宣称量化收益。
随后两路独立 warmup，正式 float→完整混合组合→完整混合组合→float，
核对同路 seed hash 重复、每个量化投影实际 `calls>0`、基线 `calls=0`。
全 identity 组合必须真实生成但量化 calls 为零，不伪造量化调用。
保存全部 PNG/NPY、逐请求快照与计时，所有正式候选图的
**relative L2 与 scaled max 均 ≤1%** 才继续；RGB/RGBA 全通道参与比较，
仍保守标记有损，不等于视觉质量或 HTTP 验收。
释放所有量化投影的原浮点基线，以模块和 weight 弱引用核实无残留；
identity 投影有意保留原权重。再完整生成并复现候选 hash，
正常关闭引擎且监督器成功后，父进程才原子发布批准 profile。

证据含 `environment.json`、`activation-calibration.json`、`samples.json`、
`comparison.json`、`released.json`、`completed.json`、`closed.json`、
`supervisor.json` 与 `published.json`。失败保留候选／拒绝证据，不发布默认批准文件。
快照 `quantization.modules` 提供实际 bits、calls/rows、source_bytes/storage_bytes；
后两者仅为量化投影持久状态口径，不是 RSS 或低峰值流式加载测量。
profile 只证明记录的输入／seed／尺寸／步数，不外推 2K、编辑、所有提示、多 seed 或长期负载。
量化投影对行数 shape-polymorphic，不依据单次校准尺寸／步数新增 HTTP 请求限制，
不改变原生默认；已有 dtype、连续布局、group64 与独立联合批准合同仍严格执行。

批准后通过显式 dynamic 与 profile 路径启用：
`QWEN_IMAGE_METAL_QUANTIZATION=dynamic QWEN_IMAGE_METAL_QUANTIZATION_PROFILE=/absolute/approved.json ./start.sh`。
准备器默认发布路径为 `~/.cache/qwen-image-metal/dynamic-approved.json`，
生产仍须显式提供路径，不会自动启用。服务运行中不会热改模块；
准备工具不启停任何 HTTP 服务。

### 实际 ANE gate + DynamicEngine down 联合校准

默认仍为 off，尚无本次真实联合通过证据。先显式离线构建共享 ANE 工具，
再在私有环境执行（不加载或重启 HTTP 服务）：

```bash
"$runtime/bin/python" -I -m vllm_omni_metal.ane.build --bridge
"$runtime/bin/python" -I prepare_quantization.py --mode joint \
  --layers 0,10,21,31 --chunks 1 --size 512x512 --seed 42 \
  --output /absolute/new-qwen-joint-evidence \
  --profile /absolute/qwen-joint-approved.json
```

准备器捕获真实8步激活，搜索实际 W4/W8 down，离线准备 gate 线性 NE 图。
以相同量化 GPU 完整 pipeline 为计时对照，候选最慢样本必须仍快至少3%；
完整原生→联合→联合→原生图片双1%与各路径同 seed hash、每个量化 down
实际调用、每个 gate 非零 ANE 命中及 live 完整投影双1%均须通过。
独立 ANE 缓存策略在 worker 内暂存/校验，不能单独构成联合批准。
释放量化 down 浮点模块/权重后重现图片，再关闭私有执行器和 engine，
supervisor 成功才原子发布最终 owned0600 联合 profile；失败不发布联合批准。
scope 绑定完整 checkpoint、两实现、ANE 源码/图身份、校准器与执行配置。
wheel 源 SHA、RECORD、fingerprint、ready 覆盖上述全部私有模块和准备器。

批准后显式启用：`QWEN_IMAGE_METAL_QUANTIZATION=joint
QWEN_IMAGE_METAL_QUANTIZATION_PROFILE=/absolute/qwen-joint-approved.json
VLLM_OMNI_METAL_ANE=auto QWEN_IMAGE_METAL_ANE_LAYERS=0,10,21,31 ./start.sh`。
不同层、请求 context/形状、无实际 ANE 命中、旧独立批准均明确拒绝，不静默 GPU 回退。


## GPU／ANE 算子接入与离线校准

`patches/qwen-metal.patch` 只为当前原生 QwenImage pipeline 绑定本目录 `metal_adapter.py`：保留官方 registry、组件、采样与 BF16，给已有 FFN/RMSNorm/interleaved RoPE 提供通用算子入口。`prepare_runtime.py` 根据共享官方 Omni commit、私有补丁、适配器和通用后端源码生成指纹，隔离构建非 editable wheel；构建依赖只进入临时构建环境，不把更高版本 setuptools 装进运行时。校验失败不写 `.ready`，不修改共享基础环境。

通用后端源码已按功能分包：diffusion 算子为 `vllm_omni_metal.diffusion.ops`，GPU 构建入口位于 `vllm_omni_metal.ops.vpipe_kernels`，ANE 运行时为 `vllm_omni_metal.ane.runtime`。准备器、安装检查与 ready 校验使用新路径，源码／资源指纹递归包含全部子包。本次仅同步源码，不代表当前运行服务或已安装环境已迁移；没有运行准备器、切换 ready／current、重启服务或启用量化／ANE。

| 路径 | 接入方式与限制 |
|---|---|
| diffusion attention | `MetalSDPABackend` → Torch MPS SDPA；不是 AR paged-attention |
| RMSNorm | 保留 Diffusers FP32 variance 和按 weight dtype 决定 cast 的合同，匹配时用 Torch MPS fused norm |
| SwiGLU | gate/up 独立输入，无额外大 `cat`；已有 vpipe AOT 优先，否则大输入可用现有 MLX bridge |
| RoPE | MPS实际分支调用`mps_rotary`尝试独立diffusion AOT；off、缺产物或布局不支持时保留原实数Torch后备；不误用NeoX表，CPU/CUDA分支不变 |
| GPU＋ANE gate | 通用 `ane.projection.DenseProjection` 分配256行 chunk；down 不变或由 GPU DynamicEngine 执行，256 MiB/4 GiB 与 live 完整投影双1%门禁不变 |
| 不相符的原生 AR 优化 | AR 分页 KV、NeoX 缓存表、add-before-RMSNorm 不直接套到 diffusion 条件前缀 KV／interleaved RoPE／LayerNorm |

部署入口`start.sh`默认将`QWEN_IMAGE_METAL_GPU=auto`和`VLLM_OMNI_METAL_ANE=off`写入launchctl plist，普通`./start.sh qwenimage`即可使用可用GPU AOT，不启用ANE。显式环境变量仍可覆盖；用`QWEN_IMAGE_METAL_GPU=off`恢复原Torch数学路径。AOT需事先显式构建，缺产物保留已有后备，不在启动/请求期编译。若显式设置ANE auto，也只消费匹配权重／pipeline scope／请求形状的已批准profile，当前没有批准的QwenImage ANE profile。`QWEN_IMAGE_METAL_ANE_LAYERS=0`仅指定候选第0层，不代表ANE已启用或全部32层协同。

只有已安装 Apple Metal compiler 时才显式准备生产 AOT：

```bash
# 从根目录运行；先确认共享构建不会中断其他模型。
backend="$(python3 prepare_backend.py)" &&
"$backend/bin/python" qwen-image-2.1-turbo/prepare_runtime.py --with-vpipe
```

`--with-vpipe` 同时准备原 vpipe kernels 和 diffusion interleaved RoPE，普通启动不编译。现已安装 Xcode 27.1（27A9275）和官方 Metal Toolchain 27A266a，`metal`／`metallib` 路径、版本、初始化及实际 `.metal → .air → .metallib` 编译链接验证通过。工具安装说明见 [Apple 官方文档](https://developer.apple.com/documentation/xcode/downloading-and-installing-additional-xcode-components)。

完整真实生成校准需事先释放资源；停止模型会中断在途请求，不能为了校准启动 `all`。启动前至少 32 GiB available，运行中至少 4 GiB available、进程树 RSS 不超过42 GiB、磁盘保留2 GiB；达到停止线只清理本次校准进程，不停止其他应用、不提高 MPS 水位。校准使用固定提示、seed42、512×512、官方8步；每路独立 warmup，正式样本正反交错，包含 encoder→DiT→VAE→CPU 图片返回，不计模型加载或 PNG 保存。离线引擎通过 `OmniDiffusionConfig.enrich_config()` 从官方 `model_index.json` 解析原生 pipeline 类名，与服务端原生类解析一致，不在通用后端硬编码模型名。无真实输出、命中计数、同模式复现、最终 FFN／整图误差证据时不能批准。

```bash
# 从根目录执行；确认可中断当前图片请求后，只停止 QwenImage。
./stop.sh qwenimage
backend="$(python3 prepare_backend.py)" &&
runtime="$("$backend/bin/python" qwen-image-2.1-turbo/prepare_runtime.py)" &&
"$runtime/bin/python" qwen-image-2.1-turbo/prepare_ane.py --mode gpu-ab \
  --output "$PWD/../vllm-omni-metal/outputs/qwenimage-gpu-ab-$(date +%Y%m%d-%H%M%S)" &&
"$runtime/bin/python" qwen-image-2.1-turbo/prepare_ane.py --mode ane --layers 0 --chunks 1 \
  --output "$PWD/../vllm-omni-metal/outputs/qwenimage-ane-$(date +%Y%m%d-%H%M%S)"
```

`--layers 0` 明确只校准第0层，不冒充全32层；在资源允许时可显式指定多层。条件编辑校准可给两条命令都加 `--image /存在的实际草图.png`，与文生图分开报告。此工具仅接受512／768／1024方形请求，未验证2K。参数并不保证峰值可容纳，守护器达到预算会失败退出。

ANE FP16 为显式近似计算，不是透明 BF16 等价替换。当前 gate-only 路径必须通过 live 完整投影及整图双1%，候选最慢全pipeline样本比对照最快样本至少快3%；不通过就拒绝联合批准。生产只读预先准备的策略，不在请求里编译、探测或调整分配。以下 full-FFN 历史证据不适用于新的 gate-only/joint 路径。

此前通用算子／运行时专项113项、非集成326项通过；私有适配MPS回归确认原始FFN不变、busy拒绝切换、准备失败保留健康旧组及幂等关闭，这些不是本次重跑结果。共享环境仍为`runtime/c076adaf5074c803`、Omni main `082eaf235cfc373322bc5f8a1a3398915446a82d`；此前overlay为`bf033dac54505e3e`，本次RoPE修复已通过准备器构建新的非editable overlay `6d492db20f717f02`，未修改共享安装或重启HTTP服务。



真实官方第0层 BF16 权重（4096→12288→4096）已完成图准备及1024行小张量测试：inner 分为10240与2048，每片执行一个256行 ANE chunk、768行 GPU；最终相对L2 **0.297%**、峰值归一误差 **0.455%**，有限输出且未触发门禁。CoreML 每个投影计划均首选 ANE；这是计划与实际调用证据，不是 Instruments 硬件 trace。最新源码与私有 overlay 重跑证据在相邻源码仓库 `outputs/qwenimage-real-ffn-20261016-final/`；输入是小张量而非模型实际激活，**未发布完整 pipeline profile**。

未构建AOT时的历史真实GPU A/B（512×512、seed42、8步；Omni main `082eaf235cfc373322bc5f8a1a3398915446a82d`）：GPU off 27.246／27.823秒，GPU auto 28.695／28.355秒；输出SHA相同、像素误差为0，但auto慢约3.6%，故保持GPU off。当时`aot_silu=false`、`aot_rotary=false`。第0层ANE协同实际执行8次调用、16个ANE chunk；整图相对L2 **0.6697%**，scaled max **12.94%**，超过1%门禁，未批准／发布profile。此前统一入口512×512文生图返回HTTP 200、PNG/RGBA、1,049,508字节，结果`/tmp/qwen-image-test.png`。历史证据：`outputs/qwenimage-gpu-ab-20261016-run6/`、`outputs/qwenimage-ane-20261016-run2/`。

工具链安装后，使用原overlay `bf033dac54505e3e` 的 `python -B -I`，直接调用已安装后端的 `build_vpipe_kernels()` 与 `build_diffusion_kernels()`，未更新共享版本。生产产物位于 `~/.cache/vllm-omni-metal/vpipe/d468e9a681021194bd360ae1fea15e2eef2f64d5ded5537f6bd818973e71ce22/` 和 `~/.cache/vllm-omni-metal/diffusion/e759c87b8abeed212590b5639a03ba625c83753baacdccf7c3a06f82fd1829aa/`。实际metallib的FP32/FP16/BF16独立输入SwiGLU与批量interleaved RoPE数值烟测通过，分别3次`aot`命中；与CPU参考的scaled max均低于1%，BF16本次小张量两算子差异均为0。

释放Docker Desktop资源后，available达到32.04GiB，完整AOT GPU A/B通过（相同overlay、512×512、seed42、8步；每路独立预热，正式off→auto→auto→off）：off为27.315443／27.215885秒，auto为27.211619／27.036432秒。中位延迟约降低0.52%，每路只有两个正式样本，不据此认定稳定提速或修改默认off。所有样本SHA一致、像素误差为0；候选每图实际命中256次`silu_mul:aot`，512次`rms_norm:torch`，未记录RoPE命中，不能将RoPE产物存在当成整图接入证明。实际生成512×512 RGBA窗台橘猫，目视符合提示。守护器returncode=0、无停止原因，最低available约7.91GiB，engine正常关闭。证据保留在相邻源码仓库`outputs/qwenimage-aot-gpu-ab-20261016-run1/`，包括环境、样本、图片、对照、守护器和关闭记录。本次为离线生成，不是统一HTTP接口验收，也未重跑ANE、2K或编辑。

RoPE未命中的根因是旧私有补丁把适配器接在CUDA专用`_apply_qwen_image21_rotary_emb`内，实际MPS调用却直接进入`_apply_qwen_image21_rotary_emb_native`。现于MPS真实分支尝试`mps_rotary`，只在auto且AOT支持时调用通用算子；off、缺产物、非连续输入仍回到原实数Torch后备，不强制复制布局。旧wheel行为回归在缺失`rotary:aot`计数处预期失败；新wheel的FP32/FP16/BF16×MPS auto/off/非连续/CPU共12例通过，另用空临时HOME确认缺AOT时逐值保留native输出。AST与Ruff format检查通过；Ruff lint仍报已有导入分组I001，未为此重排无关导入。

修复版overlay `6d492db20f717f02` 真实生成A/B（同提示、512×512、seed42、8步；独立预热后off→auto→auto→off）：off为27.324780／27.227887秒，auto为26.047730／26.002932秒，中位延迟约降低4.59%。每张候选图实际命中512次`rotary:aot`、256次`silu_mul:aot`，另有512次`rms_norm:torch`；所有样本SHA一致、像素误差为0，PNG/RGBA且目视符合窗台橘猫提示。最低available约7.54GiB，守护器正常退出、engine关闭。仅两个正式样本/路，不外推其他输入、2K、编辑或长期热负载。证据：相邻源码仓库`outputs/qwenimage-rope-aot-fixed-20261016-run1/`。新版第0层ANE重测中位耗时26.641秒，对照GPU AOT为26.044秒，scaled max仍为12.94%，未发布profile，证据为`outputs/qwenimage-rope-aot-ane-20261016-run1/`。部署默认现为GPU auto、ANE off；从根目录正常执行`./start.sh qwenimage`即可，随后运行上文模型身份与文生图curl，输出仍保存至`/tmp/qwen-image-test.png`；共享版本预检可能更新环境并停止其他模型，遵循根目录启停说明。

部署启用验收：`start.sh`在未提供两项环境变量时生成的plist默认GPU auto、ANE off，实际运行plist亦一致，CLI来自修复版overlay `6d492db20f717f02`。总代理恢复后，统一`/v1/models`聚合列表及单后端身份检查通过；上述参数真实文生图返回HTTP 200、image/png、1,049,508字节，PNG可解码为512×512 RGBA，RGB标准差84.36，目视符合窗台橘猫提示。此次文件为`/tmp/qwen-image-gpu-aot.png`和`/tmp/qwen-image-gpu-aot.headers`，公开curl示例仍写入`/tmp/qwen-image-test.*`。模型与网关当前运行；未因此验证2K、编辑或生产并发。

## 2026-10-22 本次联合安装与资源拒绝

精确官方0.32提交 `9534efe802ef79ce52af31847fb144f63d259e90` 的私有补丁已适配
新增SDPA导入及NPU rotary分支，保留MPS优先分派。正常准备器完成
`overlays/qwen-image-2.1-turbo/67f21699bb5549ba` 非 editable wheel 构建、
安装、来源/RECORD/资源/依赖校验；`.pth` 显式复用现有0.32基础依赖，不是全依赖重装。
离线子进程使用所选 Python `-I`，不受cwd/PYTHONPATH影响。

实际执行 `prepare_quantization.py --mode joint`，启动前available 27.31GiB，
低于现有32GiB门槛而拒绝，未创建模型worker、未生成图片或发布联合profile。
证据在相邻后端 `outputs/joint-ane-dynamic-20261022-222638/qwen-real/` 及同目录
`resource-before-qwen.json`／`qwen-real-launch.log`。默认仍off，未停止其它服务、
降低预算、切共享current或重启本模型；旧完整图片结果不能替代本次联合批准。

## 2026-10-23 真实联合启动复验

available达到33.24GiB后，直接用既有私有环境执行512×512、seed42、8步、
第0/10/21/31层联合校准，未跳过资源线。实际暴露并修复两项启动合同：

- 上游0.32向runner传入 `initialize_lora`；共享Metal runner现按上游合同接收，
  Torch按原时序处理，自管理MLX不调用Torch回调，不新增LoRA能力。
- Qwen的NPU专用SDPA覆写误注册到Metal，违反所选实现的继承约束。
  私有补丁只在NPU注册该覆写，保留Metal后端及原类型检查，不屏蔽异常。

正常准备器已构建并校验新私有环境 `feb09f3c459a18e5`，显式准备匹配ANE工具；
旧环境未原位修改，GitHub源码获取已恢复。真实MPS小流水线/MLX回归6项及
私有Qwen attention数值回归通过，完整非集成958项通过（5 skipped、336 deselected）。
安装器/Formula实际Python校验段、源码/RECORD/许可证检查及Ruff、Shell/Ruby语法通过。

两项修复后的完整重跑再次被启动available<32GiB拒绝，最终资源复核24.70GiB。
本轮没有生成图片、没有联合批准或部署启用，也不把加载文本权重当成图片验收。
证据在相邻后端 `outputs/qwen-joint-direct-20261023-100747*`：
最初目录为加载签名错误，`-fixed/`为覆写错误，`-attention-fixed/`为修复后的资源拒绝；
完整摘要为 `qwen-joint-direct-20261023-100747-verification.json`。

## 检查与验收边界

部署目录不保留测试脚本；回归与小算子验证在编辑期间临时执行，验证后按根目录 `AGENTS.md` 的最后清理规则删除。以下保留原验收记录，不表示清理后仍存在相应测试入口，也不是本次清理重新运行的结果。部署配置检查命令见根目录 README，真实文生图 curl 保留在上文。

此前记录：缓存/采样校验 7 项、路由 4 项、总启停 4 项通过；可选后端 2 项通过（真实 MPS fp32/bf16 RoPE+SDPA 对照 CPU，以及原生 preprocess 保留 Turbo 采样网格）。两份 Compose 配置、Shell/JS/Python 语法及独立检查镜像的 `nginx -t` 通过。这些是原验收记录，不是本次清理的检查结果。

2026-10-16 本次真实验收：修复位于相邻 `vllm-omni-metal` 的通用 `MetalSDPABackend`，直接复用上游 Torch SDPA，避免默认 `AttentionImpl.forward` 没有 OOT 分派导致预热退出；未修改模型组件、采样、精度或资源限制。经 `prepare_backend.py` 干净副本构建并激活非 editable wheel，运行时为 `~/.local/share/local-model-backend/runtime/0794b39fd3ae1c44/`，实际模块路径与非 editable 属性已检查。Apple M4 Pro / 48 GiB、macOS 26.6.2（25G83），Omni main `2176457b3d8d621e53bdbfd22a42997c8814704f`；Torch 2.13.0、Transformers 5.14.1、Diffusers 0.40.0。

本次 `./start.sh qwenimage`、不带参数的统一模型聚合列表及单模型身份检查通过；上文同参数真实文生图返回 HTTP 200、`image/png`、1,049,508 字节，27.70 秒完成。PNG 可解码为 512×512 RGBA，RGB 标准差 84.36、alpha 范围 253–255；目视为窗台橘猫、蓝天和阳光，符合提示。通用后端新增真实 MPS attention 回归 8 项和 tiny Stable Diffusion 生成 1 项通过，非集成测试 260 项通过，Ruff、依赖与部署配置检查通过。验收请求不等于 seed 字节一致性或全部能力通过。

未验证 2K、图像编辑、多图、LoRA、seed 字节一致性、取消、流式或生产并发。共享依赖导入仍有 PyAV/OpenCV/Homebrew FFmpeg 的 Objective-C 重复类警告，本次未为此改依赖。

模型许可为 **Qwen Research License**，不是普通 Qwen 模型的 Apache-2.0 假设；商用前核对官方 `LICENSE`。
