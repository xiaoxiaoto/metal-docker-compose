# Qwen-Image-2.1-Turbo · vLLM-Omni + Omni Metal

Apple Silicon 上运行上游原生 **QwenImage21Pipeline**，复用共享隔离后端的 **vLLM-Omni + vllm-omni-metal**。不使用普通 Qwen-Image pipeline、CUDA Docker、自建 HTTP adapter、CPU fallback 或 offload。官方 checkpoint 保持原精度；动态混合精度默认 off，仅消费本机真实整图校准批准的 DynamicEngine profile。

- 官方在线示例：https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/text_to_image
- 官方模型卡：https://modelscope.cn/models/Qwen/Qwen-Image-2.1-Turbo
- 对外 ID：**`qwen-image-2.1-turbo`**。下载仓库为 `Qwen/Qwen-Image-2.1-Turbo`，不把普通 Qwen-Image 示例中的模型名或 50 步参数照搬到本模型。
- 单 generation pipeline，从 `model_index.json` 解析组件与采样；无需双阶段 Omni profile，不增加无需求的 `metal-deploy.yaml`。

根目录将本模型配置为 **dynamic，空闲300秒卸载**。每次根目录启动（包括默认/gateway/lifecycle）先安全停止所有依赖模型、删除历史共享环境和私有 overlay，再重建共享后端；停止失败不删除，构建失败不回退。`./start.sh qwenimage` 每次调用本目录 `start.sh --prepare-only` 重建私有环境和 plist，首次推理才离线加载。权重、下载缓存、量化资产和日志保留；精度/量化参数以本次设置为准。详见[根目录重建合同](../../README.md#每次根目录启动前重建共享环境)。

冷模型不在聚合列表，单模型身份查询返回503且不唤醒；先查看根目录 `python3 -I runtime_mgmt/lifecycle_control.py status`，发送下文真实文生图curl，完成后复查身份。冷请求另留最多600秒加载预算；不要旁路8099推理。动态启动额外要求至少36GiB保守空闲内存，不能为了常驻H2O而提高MPS水位或放宽门槛。

## 环境与资源

Apple Silicon、macOS 15+、Python 3.11+（根控制脚本与启动引导）、ModelScope CLI。网关另需 Docker Desktop。共享运行时由 uv 托管，要求 Python `>=3.12,<3.13`；共享依赖仅由根目录 [`pyproject.toml`](../../pyproject.toml) / [`uv.lock`](../../uv.lock) 声明，以 `uv sync --locked` 非 editable 安装：vLLM/Metal `0.32.0`，Omni 固定 `9842ebc17cf1d9c6b440ecb6a79039162f68a1dd`，Omni Metal 来自 `https://github.com/xiaoxiaoto/vllm-omni-metal.git` 的 `5774df7d720b3a8c6d32035729b42c31e480f346`。共享准备不依赖相邻源码；保留官方联网预检，不随 main 自动升级，MPS 与 `pip check` 等验证通过后才激活，详见[根目录合同](../../README.md)。`prepare_runtime.py` 在 Qwen 拥有的私有 overlay 内仍分别安装当前相邻后端和打补丁 Omni 的非 editable wheel，私有 Omni 使用 `0.32.0.dev0+cpu.qwenmetal1`，保留自身来源兼容门禁。基础环境缺 diffusion／DynamicEngine API 时也不调用根准备器升级 current；不修改共享安装或其他模型环境。

vLLM 与 vllm-metal 也从官方 GitHub 源码安装，分别固定 `vllm-project/vllm` 的 `5df63a7203cdb9328d824a27b682e77c7b9928be` 与 `vllm-project/vllm-metal` 的 `6d6f2c37f8242f1ff1a336534adf6c2d2fe2936d`（v0.32.0）；由 uv 隔离构建非 editable wheel，不使用官方预编译 wheel。vLLM 的 CPU 编译形态只是 Metal 插件底座，不代表模型 CPU fallback。

模型卡的 Diffusers 直接运行示例要求开发版 Diffusers 与 Transformers ≥5.17；本部署走 **Omni 的原生实现**，不是 `diffusion_load_format=diffusers`，不能为了照搬该示例逐包升级共享依赖。原生组件能导入、小算子通过都不等于真实权重能完成生成。

ModelScope 文件清单的权重总计 **32,440,133,648 字节（30.21 GiB）**：Qwen3-VL encoder、两片 transformer 与 RGBA VAE。首次先测试 512×512、单图，不与其他大模型并驻；2K 请求须独立进行资源和质量验收。

准备及显式加载均检查物理内存至少为权重预算 +10 GiB、Metal 推荐预算至少为权重预算 +4 GiB；这些是设备能力预检，不代表测得的峰值上限，`--prepare-only` 也不跳过。下载/补全前的剩余磁盘至少完整权重 +5 GiB 检查仍保留（不完整缓存也按完整下载预算检查），不自动清理缓存。

中央扩展包的 `prepare.py` 固定选择 `~/.cache/modelscope/hub/models/Qwen/Qwen-Image-2.1-Turbo/`，完整缓存直接复用；否则下载/补全官方 `master`，排除 `assets/*` 示例媒体。静态校验组件架构、未量化配置、8 步递减有限 `sample_sigmas`、processor/tokenizer、权重索引、路径与 safetensors 数据边界，不执行缓存代码或反序列化 pickle。不自动换仓库或选择其他版本；`master` 非不可变 revision，首次下载需核对仓库与研究许可。

原生模型保留 Qwen3-VL encoder、transformer、RGBA VAE、scheduler 和官方 `sample_sigmas`，不手写采样网格。配置为 bf16、单设备 `uni`、`TORCH_SDPA`、eager、VAE slicing/tiling；不删除 unsupported 校验，不接受上游量化 checkpoint/config，不启用 offload、并行或提高 MPS 内存水位。下述独立动态接入只在官方未量化权重加载完成后应用已批准的 down 混合精度组合。


## 启停

从项目根目录执行：

```bash
./start.sh gateway  # 首次更新路由需构建网关；若网关已运行，会中断在途请求，先确认
./start.sh qwenimage
./stop.sh qwenimage # 只停止本模型，不停止网关或其他模型
```

可单独执行 `composes/qwen-image-2.1-turbo/start.sh --prepare-only`：复用完整准备流程，在原子写入 plist 并通过 `plutil -lint` 后退出，不 bootstrap 或停止已有 worker。已注册任务和端口冲突仍拒绝，非法参数非零退出。无参数直接运行本目录 `start.sh` 仍准备并加载模型；无需先加载再停止来准备动态部署。

先确认大权重下载、磁盘与内存预算，不为单模型验证启动 `all`。未提供 `VLLM_BACKEND_PYTHON` 时，本目录启动脚本也调用共享准备器，安全停用全部依赖模型、删除历史环境后重建；只恢复所选目标。`stop.sh` 仍只停止本模型，不操作网关。

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

官方草图为 3,200×1,120 RGBA、1,878,998 字节，超过当前 **64 KiB 整个请求体**上限，所以使用后端支持的 `url` 表单字段，不直接上传。当前后端下载器不跟随 HTTP 重定向，示例先用 HEAD 解析最终图片 URL，再让后端下载；签名 URL 可能过期，重试时重新执行整段，不保存或硬编码。客户端和后端均需能访问该站点；自行替换 URL 时只使用可信、可直接返回图片的 HTTPS 地址。不要手设 multipart Content-Type/boundary。

`use_kv_cache=True` 是 Diffusers 参数，当前 HTTP 编辑接口不暴露该字段；添加同名表单项不能证明已生效。原生 pipeline 根据 transformer 的 `causal_condition` 自动创建每请求 KV cache，本地官方配置为 `true`，无需额外传参。RGBA 条件图由原生预处理保留/转换，不需要在客户端安装 Diffusers；seed 与 CPU generator 参数不代表与 Diffusers 输出逐字节一致。

## 逐模型动态混合精度

模型专属扩展源文件位于 `../../extensions/vllm_omni/qwen_image_2_1_turbo/`：
`metal_adapter.py`、`quantization_adapter.py`、`joint_adapter.py` 及
`prepare_quantization.py`、`prepare_ane.py`、`prepare.py`。`prepare_runtime.py`
从该包读取并将资源私有非 editable 打包进 Omni wheel；源 SHA、Omni RECORD、
overlay 指纹和 `.ready` 同步校验。离线 CLI 也从该包路径运行。
安装/ready校验用选定Python `-I` 导入共享 `quantization.capture`、`adapter`、
`runtime`，递归后端源码指纹覆盖新模块；通用入口接通不是任意Omni模型已移植的证明。

当前私有补丁和准备器要求官方 Omni 提交 `9534efe802ef79ce52af31847fb144f63d259e90`，而共享锁文件固定的是 `9842ebc17cf1d9c6b440ecb6a79039162f68a1dd`。**两者不匹配，当前共享环境会被私有准备器拒绝，启动及校准因此受阻**；须先针对固定共享来源完成适配，不得绕过来源门禁或擅自升级共享依赖。固定版本的 pipeline 还支持独立的 step execution 路径，而现有补丁只在 `forward` 包围请求生命周期；在确认该路径的适配器生命周期和完整推理语义之前，不能仅替换 commit 声称兼容。公共合同未实现 Qwen 的 ANE 或联合模式。

公共优化默认关闭，`METAL_MODEL_OPTIMIZERS` 显式选择 `gpu_ops` 或
`dynamic_quantization`（逗号分隔）；量化仅接受 `METAL_QUANTIZATION=off|dynamic`，
dynamic 必须同时选择 `dynamic_quantization` 和设置绝对路径
`METAL_QUANTIZATION_PROFILE`。本模型的 Omni 执行器不支持公共 ANE、
streaming 或联合配置，显式选择时拒绝；不会将其他执行器的 provider 当作已支持。
缺 profile、未批准或执行身份不匹配均报错，不静默回退。

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

独立 dynamic 与历史 ANE 批准不能拼接；joint profile 不是生产入口。
scope 绑定整个本地模型 safetensors、配置、tokenizer、适配器、通用
linear/engine 源码、上游、原生采样和 GPU 路由；实际消费完整读盘哈希。
旧 schema／旧实现批准不可迁移，必须重新校准。不得将源码接入视为已批准、
已启用、无损、加速或节省全模型峰值；GPU AOT 不是 INT8。


### 真实整图离线批准

使用来源兼容的共享基础环境，不调用共享 `common_env/prepare_backend.py` 或升级 current；目前固定来源不匹配，须先解决上述阻塞。

```bash
cd composes/qwen-image-2.1-turbo
base="$VLLM_BACKEND_PYTHON"
runtime="$("$base" -I prepare_runtime.py)"
"$runtime/bin/python" -I ../../extensions/vllm_omni/qwen_image_2_1_turbo/prepare_quantization.py \
  --layers 0,10,21,31 --size 512x512 --seed 42 \
  --output /绝对路径/qwen-dynamic-独立新目录
```

已存在证据目录会拒绝覆盖；本机批准文件先在同目录完整写入、fsync，再以原子 no-overwrite
hard-link 提交，并发发布也不覆盖赢家。可用 `--profile` 指定新文件。
不需要 `--with-aot-kernels`，缺 Metal compiler 不应冒充已启用 AOT。若仅安装／验证 overlay，
准备器会真实构建私有 wheel，不触碰已安装上游；上面的校准另由明确命令执行。
安装与 pip/check_versions 校验均使用所选解释器 `-I`；后端通过该解释器的 `pip wheel`
隔离构建，Omni 则由 uv 的隔离构建环境及显式 `--python` 选择解释器。
相邻 `vllm-omni-metal` 的 `pyproject.toml`、README、LICENSE、全部 Python／native 资源
生成 manifest，参与 overlay 指纹；后端从只含 manifest 文件的干净临时副本构建 wheel，
只安装到本模型 overlay。ready 检查后端非 editable 来源、完整资源集合与逐文件 SHA、
随包 LICENSE、后端及 Omni 全部已哈希 RECORD 项；完整 manifest 保存在 `runtime.json`。
基础依赖仍通过明确 `.pth` 复用，不能把此操作称为全量依赖重装。
新建 overlay 不运行 ensurepip；pip 与基础依赖均通过明确的共享 base `.pth` 继承，准备器变化会产生新指纹，不复用旧私有 pip 环境。
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

批准后通过 `METAL_MODEL_OPTIMIZERS=dynamic_quantization,gpu_ops METAL_QUANTIZATION=dynamic METAL_QUANTIZATION_PROFILE=/absolute/approved.json ./start.sh` 显式启用；provider 组合须与批准时执行配置一致。
准备器默认发布路径为 `~/.cache/qwen-image-metal/dynamic-approved.json`，
生产仍须显式提供路径，不会自动启用。服务运行中不会热改模块；
准备工具不启停任何 HTTP 服务。

### ANE 与联合模式

公共 ANE profile 在此 Omni 执行器尚未实现，联合生产与 ANE 离线准备均明确拒绝；旧联合校准文件不能作为当前配置的生产批准。



## GPU／ANE 算子接入与离线校准

`../../extensions/vllm_omni/qwen_image_2_1_turbo/qwen-metal.patch` 只为当前原生 QwenImage pipeline 绑定包中的 `metal_adapter.py`：保留官方 registry、组件、采样与 BF16，给已有 FFN/RMSNorm/interleaved RoPE 提供通用算子入口。`prepare_runtime.py` 从该包读取模型适配器、准备器和私有补丁，根据共享官方 Omni commit及通用后端源码生成指纹，隔离构建非 editable wheel；构建依赖只进入临时构建环境，不把更高版本 setuptools 装进运行时。校验失败不写 `.ready`，不修改共享基础环境。

通用后端 diffusion 算子位于 `vllm_omni_metal.diffusion.ops`，GPU 构建入口位于 `vllm_omni_metal.ops.aot_kernels`，ANE 运行时位于 `vllm_omni_metal.ane.runtime`。源码和资源指纹递归包含子包；AOT 缓存使用 `aot/elementwise` 和 `aot/diffusion` namespace。

| 路径 | 接入方式与限制 |
|---|---|
| diffusion attention | `MetalSDPABackend` → Torch MPS SDPA；不是 AR paged-attention |
| RMSNorm | 保留 Diffusers FP32 variance 和按 weight dtype 决定 cast 的合同，匹配时用 Torch MPS fused norm |
| SwiGLU | gate/up 独立输入，无额外大 `cat`；匹配的通用 elementwise AOT 优先，否则大输入可用现有 MLX bridge |
| RoPE | MPS实际分支调用`mps_rotary`尝试独立diffusion AOT；off、缺产物或布局不支持时保留原实数Torch后备；不误用NeoX表，CPU/CUDA分支不变 |
| GPU＋ANE gate | 公共 ANE profile 尚未实现，不能启用联合模式 |
| 不相符的原生 AR 优化 | AR 分页 KV、NeoX 缓存表、add-before-RMSNorm 不直接套到 diffusion 条件前缀 KV／interleaved RoPE／LayerNorm |

部署入口默认不选择任何优化，保留原 Torch 数学路径；显式选择 `gpu_ops` 才使用此模型已接入的 diffusion GPU 算子。AOT 须事先准备，缺产物沿用算子的既有后备，不在启动或请求期编译。ANE 与 streaming 的公共 profile 接口尚未实现，不能启用。

只有已安装 Apple Metal compiler 时才显式准备生产 AOT：

```bash
# 从根目录运行；先确认共享构建不会中断其他模型。
backend="$(python3 common_env/prepare_backend.py)" &&
"$backend/bin/python" composes/qwen-image-2.1-turbo/prepare_runtime.py --with-aot-kernels
```

`--with-aot-kernels` 同时准备通用 elementwise kernels 和 diffusion interleaved RoPE，普通启动不编译；安装校验定位 `_native/elementwise.metal`、`_native/diffusion_rotary.metal` 和 `_native/THIRD_PARTY_NOTICES.md`。工具安装说明见 [Apple 官方文档](https://developer.apple.com/documentation/xcode/downloading-and-installing-additional-xcode-components)。

完整真实生成校准需事先释放资源；停止模型会中断在途请求，不能为了校准启动 `all`。启动前至少 32 GiB available，运行中至少 4 GiB available、进程树 RSS 不超过42 GiB、磁盘保留2 GiB；达到停止线只清理本次校准进程，不停止其他应用、不提高 MPS 水位。校准使用固定提示、seed42、512×512、官方8步；每路独立 warmup，正式样本正反交错，包含 encoder→DiT→VAE→CPU 图片返回，不计模型加载或 PNG 保存。离线引擎通过 `OmniDiffusionConfig.enrich_config()` 从官方 `model_index.json` 解析原生 pipeline 类名，与服务端原生类解析一致，不在通用后端硬编码模型名。无真实输出、命中计数、同模式复现、最终 FFN／整图误差证据时不能批准。

```bash
# 从根目录执行；确认可中断当前图片请求后，只停止 QwenImage。
./stop.sh qwenimage
backend="$(python3 common_env/prepare_backend.py)" &&
runtime="$("$backend/bin/python" composes/qwen-image-2.1-turbo/prepare_runtime.py)" &&
"$runtime/bin/python" extensions/vllm_omni/qwen_image_2_1_turbo/prepare_ane.py --mode gpu-ab \
  --output "$PWD/../vllm-omni-metal/outputs/qwenimage-gpu-ab-$(date +%Y%m%d-%H%M%S)"
```

条件编辑 GPU A/B 可加 `--image /存在的实际草图.png`，与文生图分开报告。此工具仅接受512／768／1024方形请求，未验证2K。公共 ANE profile 尚未实现，ANE 准备模式明确拒绝。

模型许可为 **Qwen Research License**；商用前核对官方 `LICENSE`。

