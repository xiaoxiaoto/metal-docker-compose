# 本机模型总代理

统一入口：`http://127.0.0.1/v1/`（本机 80 端口）。根目录 Docker 只运行一个 Nginx + njs 代理；Metal 模型继续在 macOS launchctl 中独立运行。

## 路由

| 请求 / model | 后端 |
|---|---|
| `GET /v1/models`（无需 model） | 汇总已就绪后端的模型，返回 OpenAI 标准列表 |
| 精确路径 `/v1/systemone`（无需 model） | H2O shim，本机 `8740` |
| 精确路径 `/v1/decisions`（必须 `model=h2o-lightning-4b`） | 经 njs 校验后转 H2O shim，本机 `8740` |
| `h2o-lightning-4b` | H2O vLLM，`8000`（OpenAI 标准接口） |
| `higgs-audio-v3-tts-4b` | Higgs Omni，`8095` |
| `yue2-3b` | YuE2 Omni，`8096` |
| `moss-transcribe-diarize` | MOSS vLLM，`8097` |
| `paddleocr-vl` | PaddleOCR-VL vLLM + Metal，`8098` |
| `qwen-image-2.1-turbo` | Qwen-Image vLLM-Omni + Omni Metal，`8099` |
| `seedvr2` | SeedVR2 7B FP16，Omni + Omni Metal，`8100` |

`GET /v1/models` 不需要 `model` 参数，并行查询七个后端，只列出实际返回且与路由配置匹配的模型；后端不可达、超时或响应异常时跳过并记录日志，全部不可用时返回空列表。每个模型保留后端的 `id`、`object`、`created`、`owned_by`，不暴露本地模型路径。带 `?model=...` 的旧查询仍按模型转发，用于单后端身份检查。

```bash
curl -fsS http://127.0.0.1/v1/models
# {"object":"list","data":[{"id":"h2o-lightning-4b","object":"model","created":1791968752,"owned_by":"vllm"}, ...]}
```

`/v1/systemone` 经本机生命周期代理转发到 H2O shim `8740`，不检查请求里的 model。精确路径 `/v1/decisions` 则必须先通过 njs 原有的请求体类型、model 和大小校验，且只支持 `h2o-lightning-4b`，再由 `model-lifecycle.json` 的固定 `path_upstreams` 转发到同一 shim；H2O 标准请求（包括 `/v1/chat/completions`、`/v1/completions`、带 model 的 `/v1/models`）仍转发到 vLLM `8000`，不经过 shim。除精确 `/v1/systemone` 外，`/v1/` 请求仍由 njs 从 JSON、multipart 普通表单字段、URL-encoded 表单或查询参数读取 `model`。模型进程、请求方法、原始路径、查询参数和请求体语义不改变；响应保留二进制音频、SSE 和后端状态码。

重复表单 model、查询/请求体 model 冲突和缺失 model 返回 400；未知模型返回 404；不支持的请求体类型返回 415；超出大小限制返回 413。通过上述校验后，已知非 H2O 模型请求 `/v1/decisions` 返回明确不支持的 400，不转发到其他模型后端；`/v1/decisions/` 返回 404，不接受末尾斜杠绕过。动态模型的推理请求等待冷启动；未准备、启动失败或超时返回 503。常驻但未启动的后端返回 502。

部署名称统一为模型卡仓库名 `/` 后的完整文本转为全小写，保留连字符等其余字符，不带作者前缀或 `:latest`。例如官方 `h2oai/h2o-lightning-4b` → `h2o-lightning-4b`；Higgs 使用实际模型卡 `bosonai/higgs-audio-v3-tts-4b` → `higgs-audio-v3-tts-4b`，不再使用手写别名。下载仓库名和缓存路径仍保留作者前缀；目录名、launchctl label 和启停简称不变。

模型名必须与后端 `/v1/models` 的 ID 完全一致。更换仓库时，同步更新 `router.mjs` 的 `targets` 和根目录 `start.sh` 的模型身份检查；不根据用户输入拼接上游地址。已有部署改名后需执行 `./stop.sh all && ./start.sh all` 重载模型和代理，会中断在途请求；调用方的 `model` 字段也须同步更新，旧名称返回 404。

## 通用 DynamicEngine 量化安装合同

通用后端全局参数 `VLLM_OMNI_METAL_QUANTIZATION=off|dynamic` 默认off，与下节
生命周期的dynamic模式无关。dynamic要求
`VLLM_OMNI_METAL_QUANTIZATION_PROFILE=/absolute/approved.json`，必须当前用户
owned0600、非符号链接、4MiB内的批准schema2，或
`{"profiles": {实际完整执行scope: profile}}` 多阶段集合；不是按模型名选profile。
YuE2、Higgs、SeedVR2、Qwen-Image分别以 `YUE2_METAL_`、`HIGGS_METAL_`、
`SEEDVR2_METAL_`、`QWEN_IMAGE_METAL_` 的 `QUANTIZATION`／`QUANTIZATION_PROFILE`
参数独立覆盖全局；显式模型off覆盖全局dynamic。旧W8-only参数不再作为运行入口，
历史策略/音频/权重/profile资产保留，不自动迁移批准。

共享及私有ready校验使用选定Python `-I` 导入
`quantization.capture`／`adapter`／`runtime`；共享指纹覆盖配置、锁文件及校验源码，Git commit 固定远端实现；私有 overlay 保留各自源码指纹。
四模型私有校准器由正常Omni wheel分发、校验源码SHA与RECORD归属，离线CLI仍从
各部署源码执行。普通nn.Linear、TP1未量化vLLM全投影、Diffusers未注册components
已接通用执行适配；MLX自管理对象必须提供完整scope与apply hook，缺合同明确错误，
不静默转CPU/浮点。并非本表所有模型或任意Omni已移植。
FP16/BF16保持源dtype，FP32只identity；所有identity也合法，但不是量化收益。
四模型实际候选、完整真实任务质量门禁见各自README；安装/小图通过不能替代真实质量、
速度或冷启动峰值验证。本次仅改安装资源/说明，不执行准备器、启用部署、修改资产、
切换ready/current、重启服务或发布远端。

## 通用模型生命周期

`model_lifecycle.py` 是独立于 Metal、vLLM、Omni 和具体模型的标准库 HTTP 生命周期代理。每个服务只提供固定上游地址、argv 启停命令、就绪检查、模式和超时；不含模型名分支、量化或采样逻辑。`lifecycle_launch.py` 是本部署的 macOS launchctl 适配器，不是通用核心的依赖。任意本机 HTTP 推理后端可使用相同配置合同接入。

部署配置为 `model-lifecycle.json`，本次指定：

| 模型 | mode | 空闲卸载 |
|---|---|---|
| `h2o-lightning-4b` | `resident` | 不自动卸载 |
| `higgs-audio-v3-tts-4b` | `dynamic` | 300 秒 |
| `yue2-3b` | `dynamic` | 300 秒 |
| `moss-transcribe-diarize` | `dynamic` | 300 秒 |
| `paddleocr-vl` | `dynamic` | 300 秒 |
| `qwen-image-2.1-turbo` | `dynamic` | 300 秒 |
| `seedvr2` | `dynamic` | 300 秒 |

每项服务可单独设 `mode`、正数 `idle_timeout`、`startup_timeout`（默认600秒）、`request_timeout`（加载完成后的完整请求预算，默认1800秒；Higgs设为600秒）。`upstream` 与可选 `path_upstreams` 必须为固定 loopback HTTP 地址；`start_command` / `stop_command` 为 argv 数组，不执行 shell，也不拼接请求输入。`ready_path` 默认 `/health`；本部署用 `/v1/models` 和 `ready_model` 核对身份。通用核心可直接运行：

```bash
python3 -I model_lifecycle.py --config /absolute/path/services.json --port 18100
```

- `resident`：只转发请求；按现有显式脚本启动，不自动加载或卸载。
- `dynamic`：首次推理触发加载；同模型并发请求只加载一次，不同模型的冷启动串行。完整响应（含 SSE）结束后开始空闲计时；排队、加载、推理、响应传输中均保活。已发出但无法确认结束的上游请求会标记 `pinned`，禁止自动卸载，需人工确认后处理。
- 只卸载管理器本次成功启动的模型；已在运行的外部服务 `owned=false`，不会自动接管或停止。模型名称、精度、私有 overlay 和生成参数均沿用准备好的 plist，不执行请求期准备、下载、升级或编译。
- `/health`、聚合模型列表和单模型列表查询不会触发加载或延长 TTL。冷模型不在聚合 `/v1/models` 中；单模型查询冷模型返回503，这不代表模型不存在。冷启动前使用生命周期状态检查配置；首次推理后再执行原有模型身份 curl。
- 只对经过总代理/生命周期代理的请求提供保活保证。不要直连动态模型端口推理；旁路请求不计数，不能保证不会遇到空闲卸载。
- 启动命令必须是有界、事务式的启动器：非零退出前清理自己创建的资源，不停止外部服务。命令须在独立120秒事务预算内完成；模型加载就绪由 `startup_timeout` 另行控制。HTTP等待超时不会杀掉正在确认bootstrap结果的启动器；后台收取成功结果后回收自己启动的模型。停止命令最多60秒，管理器退出预算覆盖全部服务的串行回收。

每次执行根目录 `./start.sh [目标]`（包括默认、`gateway` 和 `lifecycle`）都重建共享环境：先联网版本预检，再原子关闭生命周期请求入口并安全停止所有依赖模型，删除历史共享环境和私有 overlay，然后从空目录安装、验证并激活。停止失败或仍有引用环境的进程时拒绝删除；构建失败不回退旧环境，相关模型保持不可用。权重、下载缓存、wheel/量化资产和日志保留。所选动态目标每次重新运行 `--prepare-only` 生成匹配新环境的 plist，保持冷状态；HTTP 冷启动仍只加载已准备部署，不下载或重建。

```bash
./start.sh gateway   # 停模型、删除历史环境并重建共享后端，然后启动管理器和网关
./start.sh yue2      # 同样重建共享后端，再准备 YuE2 私有环境和 plist；不加载权重
python3 -I lifecycle_control.py status
# 首次实际推理使用下文对应模型的 curl；冷启动需在 --max-time 中另留最多600秒
```

控制器将通用核心、适配器和配置复制到 `~/Library/Application Support/model-lifecycle/`，以 `com.local.model-lifecycle` 运行，监听 `127.0.0.1:18100`；日志位于 `~/Library/Logs/model-lifecycle*.log`。启动前核对现有任务和端口；配置/代码变化时拒绝静默重启。应用配置需先确认无在途请求，再 `./stop.sh lifecycle`、`./start.sh gateway`。管理器停止会回收其持有且空闲的动态服务，外部服务不受影响；在途请求拒绝停止。

本次 H2O Decisions 的 `path_upstreams` 修改只更新部署文件，**不会执行生命周期管理器、模型、shim 或网关的启动/重载**。配置被复制安装后才生效，当前运行实例不会热读源码目录中的 JSON。以后应用时按上一段先确认并排空在途请求，再停止生命周期管理器并启动 gateway；这会影响总代理可用性，网关重建或模型/shim 重启也可能中断在途请求，不能在有请求时直接操作。

从已运行的显式部署切到动态时：先确认该模型无在途请求，用原 `./stop.sh <目标>` 停止，再经总代理请求；下一次由管理器启动才归其所有。不修改或自动停止当前服务。H2O保持常驻，shim仍由原Compose管理。

冷启动额外检查保守空闲内存与磁盘：普通模型至少16GiB、Qwen-Image至少36GiB、SeedVR2至少24GiB，plist所在盘至少2GiB。可用内存按系统 `memory_pressure -Q` 与物理内存估算，可能保守拒绝；不主动驱逐其他模型或提高MPS水位。此检查不等于生成峰值保证；单纯串行加载也不能保证多模型并驻可行。SeedVR2保留1800秒启动等待，未准备时不会因配置动态就自动下载。

2026-10-22本次验证：14项真实子进程/HTTP生命周期回归通过，覆盖并发冷启动、TTL卸载/重载、常驻/外部服务保护、SSE/断连保活、超时pin、启动超时收尾和显式卸载后重载。隔离Nginx+njs→生命周期代理→HTTP夹具链路、唯一临时launchctl小服务的完整启停、控制器复制安装/幂等启动/指纹变化拒绝/原子drain停止均通过；Ruff、Shell/Node语法、Compose配置和原始Nginx配置检查通过。临时夹具不是模型推理验收：未加载真实权重、未切换现有80端口网关或模型服务、未验证真实模型冷启动延迟/统一内存回收峰值。Metal插件与其安装资源未修改，动态配置不等于已迁移当前运行服务。

2026-10-22 自动部署入口变更：根目录启动缺少 plist 的动态目标会自动准备，不再要求先手动部署。隔离回归通过17项根入口场景及48项实际模型启动脚本场景，覆盖六模型首次准备/复用、无效或悬空链接 plist 不覆盖、准备失败、参数/端口/注册任务拒绝、plist 实际生成和系统校验，以及 Higgs/YuE2 低可用内存下只准备不加载；依赖使用夹具，不代表真实模型推理通过。实际执行 `./start.sh yue2` 已进入自动准备，但官方 vLLM/Metal 最新为0.32.0、当前验证组合为0.31.0，版本门禁明确中止，未下载/加载 YuE2、未完成音频验收；需要另行升级并验证兼容组合，不能跳过版本检查。

2026-10-22 YuE2 本次实际修复：0.32已安装后，`./start.sh yue2` 可登记dynamic，
但首次歌曲请求因通用后端仍导入旧tensor bridge返回HTTP500。已在相邻源码修复
桥接路径、分页attention返回值及native launcher合同，构建非editable wheel，
通过已有显式wheel入口安装到YuE2私有overlay `73cdca6cd88d4c33`，复用共享
`0e63615c03ef50e1`依赖；不切共享current、不停止其他模型。官方核心预检确认
vLLM/Metal仍为0.32.0；Omni main已前移，本次仅修复已准备的部署，不迁移共享Omni。
原命令再次通过，统一入口聚合/单模型身份均含`yue2-3b`；两次真实200帧、
seed42请求均HTTP200，分别39.583秒（含冷启动）/19.883秒，WAV字节一致。
输出48kHz双声道PCM16、7.999秒、有效非静音，`X-Audio-Truncated:true`：
这是短输出验收，不是完整歌曲或听感/歌词质量通过。文件位于
`/tmp/yue2-fixed-test.wav`及`.headers`，可用`afplay /tmp/yue2-fixed-test.wav`试听。
后端真实MPS相关回归362项、随机Llama/Qwen3生成8项、非集成882项通过；
完整打包/RECORD、私有MPS自检与依赖校验通过。复测示例仍使用下方YuE2 curl；
模型空闲300秒后会卸载，下一条推理自动重新加载。
通用修复与回归已提交相邻仓库：`062ce57`，未推送；原有无关工作区改动保留。

## 启停

先启动 Docker Desktop，在本目录执行：

```bash
cd docker-compose
./start.sh          # 默认也先停模型、删历史环境并重建，再启动管理器和总代理
./start.sh gateway  # 同上；更新代理配置时使用
./start.sh h2o      # 重建共享环境 + 总代理 + H2O 模型与 shim
./start.sh higgs    # 重建共享环境 + 总代理 + 准备动态 Higgs
./start.sh yue2     # 重建共享环境 + 总代理 + 准备动态 YuE2
./start.sh moss     # 重建共享环境 + 总代理 + 准备动态 MOSS
./start.sh paddleocr # 重建共享环境 + 总代理 + 准备动态 PaddleOCR-VL
./start.sh qwenimage # 重建共享环境 + 总代理 + 准备动态 Qwen-Image
./start.sh seedvr2   # 重建共享环境并准备 SeedVR2；不加载权重
./start.sh all      # 仅重建一次共享环境，再加载 H2O、准备所有动态目标
./start.sh lifecycle # 同样重建共享环境，然后仅启动管理器
./start.sh --help
```

```bash
./stop.sh          # 默认只停止总代理
./stop.sh gateway  # 同上，不停止模型
./stop.sh h2o      # 只停止 H2O 模型及 shim
./stop.sh higgs    # 只停止 Higgs
./stop.sh yue2     # 只停止 YuE2
./stop.sh moss     # 只停止 MOSS
./stop.sh paddleocr # 只停止 PaddleOCR-VL
./stop.sh qwenimage # 只停止 Qwen-Image-2.1-Turbo
./stop.sh seedvr2   # 只停止 SeedVR2
./stop.sh lifecycle # 安全停止管理器；有在途请求则拒绝
./stop.sh all      # 停止总代理及全部模型
./stop.sh --help
```

Higgs / YuE2 / MOSS 已移除各自的纯 Nginx Compose，不再使用原来的三个对外代理端口。它们的启停脚本不再要求 Docker，也不操作总代理。H2O shim 有业务逻辑，仍由 `h2o-lightning-4b/docker-compose.yml` 管理，映射为 `127.0.0.1:8740:8741`，不占用总代理端口。它的启动/停止脚本仍管理 H2O 模型和 shim。

每次根目录启动都会先停用所有依赖模型并删除历史共享/私有环境，随后只恢复指定目标；`all` 在同一次共享重建后按原顺序处理全部目标，任一步失败停止后续操作。resident 显式加载并在最后校验聚合列表；dynamic 每次调用子目录 `start.sh --prepare-only` 重建所需私有环境和 plist，不加载权重。仅执行 `gateway` / `lifecycle` 不恢复模型部署，旧 plist 可能引用已删除的私有环境；使用模型前应执行对应目标，或使用 `all` 一次准备全部。HTTP 冷启动、健康、列表和配置登记均不替代真实推理验收。

### 每次根目录启动前重建共享环境

根目录所有有效启动目标均调用 `prepare_backend.py`，不再复用已有 `.ready` 环境；帮助、非法参数和前置检查失败不执行删除。直接调用准备器也执行同一重建合同。HTTP 动态冷启动不调用准备器。

后端准备器读取 macOS 系统 HTTP/HTTPS 代理并传给 Git、uv 和 pip；已有同名大小写代理环境变量（包括空值）优先，不硬编码端口、不关闭 TLS 校验。准备完成前统一入口可能不可用，应等待根目录命令成功后再执行身份检查和真实推理 curl。

共享后端的 Python 版本、直接依赖与来源统一声明在根目录 `pyproject.toml`，完整运行依赖由 `uv.lock` 锁定，`.python-version` 选择 3.12。需预先安装 uv 0.11.7+；引导准备/版本校验脚本的 `python3` 需 3.11+（标准库 `tomllib`），共享后端固定为 **uv 托管的 CPython 3.12**，匹配 Metal wheel 的 cp312 ABI，不选系统、Homebrew 或 Conda 解释器。准备器从 `requires-python` 读取版本范围，先安装/发现托管解释器，再检查锁文件是否与声明一致；这些步骤失败均不停止现有服务。uv 的托管 Python 不属于共享环境清理范围，不要在依赖环境使用期间执行 `uv python uninstall`。

四个核心包均从 GitHub 固定 commit 的源码安装，来源统一在 `pyproject.toml` 声明，由 `uv.lock` 锁定解析结果：vLLM 来自 [vllm-project/vllm](https://github.com/vllm-project/vllm) 的 `5df63a7203cdb9328d824a27b682e77c7b9928be`，使用该 commit 的官方源码归档并锁定 SHA256，避免抓取整个 Git 历史；Metal 来自 [vllm-project/vllm-metal](https://github.com/vllm-project/vllm-metal) 的 `6d6f2c37f8242f1ff1a336534adf6c2d2fe2936d`（均 v0.32.0）；Omni 来自官方仓库 `9842ebc17cf1d9c6b440ecb6a79039162f68a1dd`；Omni Metal 来自 [指定仓库](https://github.com/xiaoxiaoto/vllm-omni-metal) 的 `5774df7d720b3a8c6d32035729b42c31e480f346`。uv 隔离构建非 editable wheel，不使用这些包的官方预编译 wheel，也不读取相邻源码仓库或运行其安装脚本。vLLM 使用上游 CPU 构建目标并禁用预编译扩展复用，它是 Metal 插件底座，不代表模型 CPU fallback。模型私有补丁/overlay 继续由各模型准备器维护，不并入基础依赖。

源码构建还需完整 Xcode、macOS SDK 26.2+、可用 Metal/clang++ 编译器和 Rust/Cargo，准备器在停止服务前检查工具链。`vllm-metal` 的 wheel 打包不会自动编译原生扩展，因此安装后由 `prepare_native.py` 调用上游构建接口，准备 MLX/MPS launcher 和 Metal 库，并将真实产物的 SHA256/大小登记到安装包 RECORD，维持 H2O 私有 overlay 的完整性合同；登记前后均核对原 wheel 源码，不能通过重算记录掩盖源码变更。随后构建 Omni Metal 的 Torch-stream launcher；这些操作只发生在准备阶段，不在请求期间编译。

锁定范围仅为 macOS arm64。vLLM 的 CPU 元数据还包含不适用于本机的 Linux x86_64 Triton 直接 URL，uv 在平台裁剪前会尝试读取；`tool.uv.override-dependencies` 仅将这条非目标平台依赖转换为带 Linux x86_64 marker 的版本约束，避免无关下载。不修改安装包依赖元数据、不排除 Mac 运行依赖，仍执行完整 `pip check`；以后扩展平台时须重新验证该规则。

升级须显式修改 `pyproject.toml` 的版本/来源 commit，再执行 `uv lock --python 3.12 --managed-python` 并验证新组合；仅刷新锁文件不代表已部署。启动只执行锁文件一致性检查和 `uv sync --locked`，不改锁、不跟随 main 升级，也不使用根目录 `.venv` 作为模型后端。不要用普通 `uv sync` 代替部署入口，它不会执行安全停机、MPS 构建验收或发布 `current`。

1. 获取共享准备锁，联网查询官方 vLLM / Metal 最新稳定标签与 Omni main commit，保留兼容版本门禁。当前核心为 vLLM/Metal `0.32.0`、Omni/Omni Metal `0.32.0.dev0`；新核心超出配置的版本组合直接失败，不删除旧环境或混装。Omni main 前移只报告，不替换配置中固定的 commit；停止服务前检查 uv 托管 Python 和锁文件一致性。
2. 原子 drain 并停止生命周期管理器；有在途、加载或未确认完成的请求时拒绝。随后按逆序执行全部模型停止入口，包括 H2O shim；任一步失败不删除环境。
3. 核对 launchctl 任务退出、无残留进程引用环境后，仅删除 `~/.local/share/local-model-backend/runtime/<16位指纹>/`、`overlays/<模型>/<16位指纹>/` 和 `current` 链接。拒绝符号链接环境及异常所有权；其他资产、权重、下载缓存和日志保留。
4. 设置 `UV_PROJECT_ENVIRONMENT` 指向当前指纹的空目录，以 `uv sync --locked --no-dev --no-install-project` 从固定源码安装完整锁定依赖；强制 uv 托管解释器，核对新环境的基础解释器与预检选择一致。源码包由 uv 隔离构建非 editable wheel，不修改本地源仓库、Homebrew 或系统 Python。
5. 显式准备 Metal 原生扩展/库与 Omni Metal 的 Torch-stream attention launcher，再以选定 Python `-I` 执行 `check_versions.py`、`check_backend.py` 和 `pip check`，验证源码来源/commit、模块、资源、CPU 数值合同、MPS、非 editable 路径及 launcher。配置/锁文件/校验源码参与指纹，全部通过且指纹未变才写 `.ready` 与 `backend.json` 并原子激活 `current`。失败不恢复已删除环境；下次执行从空目录重建。

2026-10-27 uv 托管 Python 迁移验证：缺少 uv、解释器安装失败两条隔离回归均确认不停止服务、不删除旧环境或切换 current；临时目录真实创建 uv 虚拟环境，确认使用托管 CPython 3.12.9、原生 arm64，基础解释器路径与生成的 backend.json 一致。Python AST 与启停 Shell 语法检查通过。未执行实际共享后端重建、依赖安装、模型启动或统一入口推理，当前部署不会因本次代码修改自动切换解释器。

同日集中依赖/源码安装验收：`uv.lock` 锁定229个包（含根虚拟项目），四个核心包均从上述固定源码构建非 editable wheel；全新临时环境安装228个依赖，未继承当前部署的 site-packages。以独立临时 HOME 重新编译 Omni launcher，避免复用现有运行时缓存。版本/源码来源、MLX Metal 库、基础 MPS 扩展、Omni MPS launcher、CPU profile/diffusion 数值合同、完整 `pip check` 均通过；真实 H2O `installed_resources()` 接受固定 Git 来源并核对原生 RECORD，MPS 平方计算得到 `[1.0, 4.0, 9.0]`。8项隔离回归、Python AST、Shell、Node、两份 Compose 配置及已准备缓存下的 `uv lock --check --offline` 通过。

首次抓取/并发下载曾发生 TLS 中断；本次成功验证保留系统代理和 TLS 校验，使用 `UV_SYSTEM_CERTS=true`、`UV_CONCURRENT_DOWNLOADS=1`，源码编译限制为 `MAX_JOBS=2`。基础 Metal MPS 与 Omni MPS 的 pybind 类型不能在本次验收中混合注册，因此按独立执行器分别在子进程加载；**未验证同进程混合执行**。导入仍有 PyAV/OpenCV/Homebrew FFmpeg 的 Objective-C 重复类警告，未在本任务中处理。没有执行实际根目录重建、模型启动、统一入口推理或多模型质量验收，没有切换 `current`；临时环境和验证脚本在结束后清理，不能将本次小图/安装结果当作模型推理通过。应用时仍需先确认可中断请求，再运行所需 `./start.sh <目标>`，按下方对应模型的身份与真实推理 curl 验收。

单目标命令只恢复该目标，其余模型保持停止；动态目标的私有环境和 plist 随准备重新生成，精度/量化参数按本次环境变量与脚本合同选择，不承诺保留旧 plist 的手工设置。需要全部部署时执行一次 `./start.sh all`，不要连续执行多个根目录启动命令并期待前一个模型仍在运行。

2026-10-26 本次强制重建验证：7项隔离回归通过，含重复执行替换已ready环境、停止失败保留环境、构建失败不激活、真实目录删除/资产保留、符号链接拒绝和真实残留子进程保护；默认/gateway/lifecycle及帮助/非法参数5项根入口夹具通过，Shell、Node、Compose与Python AST检查通过。两次真实 `./start.sh h2o` 均完成安全停止、清理目标环境和新建Python/pip，随后因uv下载官方vLLM wheel连接超时退出，显式系统代理重试也失败；未激活current、未启动模型或完成真实推理。下载缓存可复用不等于复用旧虚拟环境。

共享后端及模型私有适配运行时统一置于无空格目录 `~/.local/share/local-model-backend/`：基础环境在 `runtime/<fingerprint>/`，YuE2、MOSS、QwenImage、SeedVR2 覆盖环境在 `overlays/<model>/<fingerprint>/`。`current` 指向通过验证的共享基础环境，避免 TorchInductor 将带空格的库路径拆成编译参数。迁移通过现有安装流程重新创建环境，不直接移动虚拟环境，也不修改 PyTorch 或关闭编译。旧的 `~/Library/Application Support/local-model-backend/`、`~/Library/Application Support/yue2-3b/runtime/` 和 `~/Library/Application Support/moss-transcribe-diarize/runtime/` 保留、不再使用；模型 plist、工作目录和日志仍位于各自原位置。首次迁移会停止已注册模型并中断在途请求，执行 `./start.sh h2o` 只恢复 H2O；其他目标在下次启动时使用新后端。

`all` 不再同时加载全部模型：当前只有H2O常驻，其余等待实际推理；已被动态请求加载的多个模型仍可能暂时并驻，受各自TTL和资源门禁约束。根目录 `stop.sh` 无参数只停止总代理，管理器继续空闲回收；`stop.sh lifecycle` 原子关闭请求入口后停止管理器，有在途请求则拒绝。`stop.sh all` 在管理器拒绝停止时不会继续杀模型。单模型停止先经管理器处理：自有动态模型在空闲时卸载并更新状态，有在途或未确认结束任务时拒绝；外部服务沿用原显式停止脚本，仍可能中断旁路请求。配置仍为dynamic时，下一次推理可重新加载。持续禁用服务应关闭入口并调整配置，而不是只停止worker。

## 接口调用示例

以下命令均使用统一入口 `http://127.0.0.1/v1/`，无需鉴权。H2O 需先手动下载官方模型，再在根目录启动 `./start.sh h2o`；本次只升级代码，不下载权重、不启动/重载服务。其他目标在根目录启动相应目标即可自动准备缺少的部署。resident等待加载完成；dynamic准备并登记后保持冷状态，第一条推理会冷启动。以下模型身份查询在冷状态可能返回503，应先查看 `python3 -I lifecycle_control.py status` 确认配置，首次推理完成后再查聚合与单模型身份。示例推理超时主要覆盖热推理；冷请求需额外加启动预算（通常600秒，SeedVR2为1800秒），不要将客户端超时后仍运行的任务当作空闲。

### 总代理健康检查

```bash
curl --fail-with-body --show-error --connect-timeout 5 --max-time 10 \
  'http://127.0.0.1/health'
```

返回 `{"status":"ok","scope":"gateway"}` 只表示总代理存活，不表示各模型健康。各节先检查模型身份，再发送实际推理请求。

### H2O：类型化决策

来源：[H2O.ai 官方模型卡](https://huggingface.co/h2oai/h2o-lightning-4b)。当前官方权重为 **v1.3.1**，本机权重、配置与新版 shim/tooling 统一固定为 main commit [`e7b32490c9d2a994bd75a4d1ea7e45716080175e`](https://huggingface.co/h2oai/h2o-lightning-4b/tree/e7b32490c9d2a994bd75a4d1ea7e45716080175e)，不单独下载 tag 的旧 tooling。规范对外模型 ID 仍为 `h2o-lightning-4b`。

模型由用户通过 Hugging Face 手动下载，启动器不自动下载权重。默认读取 HF 标准缓存 `~/.cache/huggingface/hub/models--h2oai--h2o-lightning-4b/snapshots/<上述 commit>/`，支持 `HF_HUB_CACHE`、`HF_HOME`、`XDG_CACHE_HOME`；可通过 `H2O_MODEL_DIR` 指定 `hf download --local-dir` 的完整目录。旧魔搭镜像缓存不删除、不自动选择。缺少发布文件时手动执行（主权重约9.32GB）：

```bash
hf download h2oai/h2o-lightning-4b \
  --revision e7b32490c9d2a994bd75a4d1ea7e45716080175e \
  model.safetensors config.json generation_config.json tokenizer.json tokenizer_config.json \
  chat_template.jinja merges.txt vocab.json preprocessor_config.json video_preprocessor_config.json \
  serve_config.json h2o_lightning_shim.py assets.py box.py pooled_readout.py example_receipt.png
# 离线检查完整发布；不会下载、加载或升级后端
python3 h2o-lightning-4b/prepare.py --check-model
```

`h2o-lightning-4b/model-release.json` 固定必需文件的官方大小和摘要，包含完整权重 SHA256；准备器流式读取，不反序列化或执行模型代码。缺少新版文件、旧缓存、混用不同 revision 或额外根目录权重会明确失败，且在共享后端升级前检查。HF 快照的同仓库 blobs 相对链接也会保留到只读容器挂载；详细下载、运行合同与未来升级命令见 [H2O 部署说明](h2o-lightning-4b/README.md)。

**部署基于 vLLM Metal，支持 two-way、one-way 与 BOX 的原生补丁路径。** `h2o-lightning-4b/patches/vllm-metal-bidirectional.patch` 固定官方 Metal `v0.32.0` commit `6d6f2c37f8242f1ff1a336534adf6c2d2fe2936d`；准备器先 `git apply --check`，再构建非 editable wheel 到 H2O 私有 overlay，不修改基础安装、共享 `current`，不引入社区 facade 或其他推理框架。shim 默认双向，保留官方启动探针；显式 `causal` 切单向，`type=box` 开启同引擎因果图像定位生成。安全部署关闭整个引擎的 prefix cache/chunked prefill，整段预算40960；one-way 也不复用前缀缓存。模型 BF16 与真正 FP32 head 合同不降低，详细限制见模型 README。

固定发布已完成文件校验，官方基础后端及私有 patched wheel 已构建验证。真实模型加载暴露的 FP32 head 探针问题已修复：通过模型原生 forward 捕获 head 输入验证投影，不再把需要原生 cache/位置初始化的 backbone 当作独立 split 路径调用。真实 Qwen3.5 FP32 head 加载及6项原生上下文/拒绝路径烟测通过；统一接口和官方 three-probe 的最终状态见模型 README，不以离线检查代替推理验收。

**Two-way：SystemOne choice / noul / score。** `/v1/systemone` 无需提供 `model`，请求使用 `state` 和对象形式的 `questions`。下例询问优先级、是否影响所有客户与紧急程度，显式指定 `attention_mode=bidirectional`；改为 `causal` 即可比较 one-way，省略默认双向。

```bash
set -o pipefail
curl --fail-with-body --show-error --connect-timeout 5 --max-time 10 \
  'http://127.0.0.1/v1/models'

curl --fail-with-body --show-error --connect-timeout 5 --max-time 10 \
  --get 'http://127.0.0.1/v1/models' \
  --data-urlencode 'model=h2o-lightning-4b'

curl --fail-with-body --show-error --connect-timeout 5 --max-time 1800 \
  'http://127.0.0.1/v1/systemone' \
  --header 'Content-Type: application/json' \
  --data-binary '{
    "state": "工单 4471：结账页面对所有客户返回 HTTP 500，已持续 40 分钟，没有订单成功。严重等级：P1 为营收路径完全不可用，P2 为降级但仍可用，P3 为外观问题。",
    "attention_mode": "bidirectional",
    "effort": "fast",
    "questions": {
      "priority": {
        "type": "choice",
        "instructions": "根据严重等级定义，这个工单应分配哪个优先级？",
        "criteria": {
          "p1": "营收路径完全不可用",
          "p2": "降级但仍可用",
          "p3": "外观问题"
        }
      },
      "all_users": {
        "type": "noul",
        "instructions": "该问题影响所有客户。",
        "criteria": {"true": "影响所有客户", "false": "仅影响部分客户"}
      },
      "urgency": {
        "type": "score",
        "instructions": "这个工单有多紧急？",
        "criteria": ["低", "中", "高"]
      }
    }
  }' | \
  python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin), ensure_ascii=False, indent=2))'
```

成功需 HTTP 200、`model=h2o-lightning-4b`、`attention_mode=bidirectional`、`probability_source=native` 与有效 `usage`；官方 three-probe 必须通过，不能绕过503。`answers` 是以问题名为键的对象：`answers.priority.choice` 应为 `p1`，`answers.all_users.noul` 是 `[0,1]` 内的真假概率，`answers.urgency.score` 是 `[0,2]` 内的等级期望值。choice 与 score 的概率分布应非负且归一化；不要把这些字段当聊天文本。JSON 中文解码仅改变展示，不修改字段或数值，`pipefail` 保留 curl 失败状态。

**One-way：官方 OpenAI Decisions 接口。** 精确 `POST /v1/decisions` 必须提供 `model=h2o-lightning-4b`，经原模型/类型/大小验证后转到 shim `8740`。下例显式 `attention_mode=causal`，也可切 `bidirectional`；请求使用 `input` 和数组形式 `questions`，`predicate` 对应真假概率、`choice` 使用 `choices`、`score` 使用 `levels`，不能直接复制 SystemOne 的请求结构。BOX 使用后面的 SystemOne 示例，不属于此 Decisions 问题类型集合。

```bash
set -o pipefail
curl --fail-with-body --show-error --connect-timeout 5 --max-time 10 \
  'http://127.0.0.1/v1/models'

curl --fail-with-body --show-error --connect-timeout 5 --max-time 10 \
  --get 'http://127.0.0.1/v1/models' \
  --data-urlencode 'model=h2o-lightning-4b'

curl --fail-with-body --show-error --connect-timeout 5 --max-time 1800 \
  'http://127.0.0.1/v1/decisions' \
  --header 'Content-Type: application/json' \
  --data-binary '{
    "model": "h2o-lightning-4b",
    "attention_mode": "causal",
    "input": "结账页面对所有客户返回 HTTP 500，没有订单成功。支付系统负责结账，网页团队负责页面外观，支持团队负责咨询。",
    "questions": [
      {
        "type": "predicate",
        "name": "outage",
        "instructions": "这是影响所有客户的完整故障。"
      },
      {
        "type": "choice",
        "name": "team",
        "instructions": "哪个团队应该负责处理？",
        "choices": [
          {"value": "payments", "description": "支付系统团队"},
          {"value": "web", "description": "网页外观团队"},
          {"value": "support", "description": "客户支持团队"}
        ]
      },
      {
        "type": "score",
        "name": "urgency",
        "instructions": "这个工单有多紧急？",
        "levels": [{"label": "低"}, {"label": "中"}, {"label": "高"}]
      }
    ]
  }' | \
  python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin), ensure_ascii=False, indent=2))'
```

成功需 HTTP 200、`model=h2o-lightning-4b` 和有效 `usage`。新版 Decisions 的 **`answers` 是数组**，按每项的 `name` 对应问题：`outage` 项的 `probability` 在 `[0,1]` 内，`team` 项的 `choice` 应为 `payments`，`urgency` 项的 `score` 在 `[0,2]` 内；choice/score 的 `probabilities` 也是数组，应为有效分布。它不同于 SystemOne 的 `answers.<问题名>` 对象，也不是 Chat Completions 的 `choices`；此接口不返回 SystemOne 的顶层 `attention_mode` 或 `probability_source`。JSON 在终端展示，不保存文件；以上示例尚未作真实新版权重推理验收。

**按 model 路由的 OpenAI 标准推理模式：** 使用 `/v1/chat/completions`，通过 `model` 选择 H2O，使用标准的 `messages` 字段；请求直接转发到 vLLM `8000`，不使用 `state` / `questions` 或末尾斜杠的变通路径。

```bash
set -o pipefail
curl --fail-with-body --show-error --connect-timeout 5 --max-time 10 \
  'http://127.0.0.1/v1/models'

curl --fail-with-body --show-error --connect-timeout 5 --max-time 10 \
  --get 'http://127.0.0.1/v1/models' \
  --data-urlencode 'model=h2o-lightning-4b'

curl --fail-with-body --show-error --connect-timeout 5 --max-time 1800 \
  'http://127.0.0.1/v1/chat/completions' \
  --header 'Content-Type: application/json' \
  --data-binary '{
    "model": "h2o-lightning-4b",
    "messages": [
      {
        "role": "system",
        "content": "你是决策引擎。读取工单并选择一个优先级，仅回复选项字母。"
      },
      {
        "role": "user",
        "content": "工单：结账页面对所有客户返回 HTTP 500，没有订单成功。应该按哪个优先级处理？\nA) P1：营收路径完全不可用，需要立即处理\nB) P2：降级但仍可用，可以稍后处理\nC) P3：仅外观问题，不影响业务"
      }
    ],
    "temperature": 0,
    "max_tokens": 64,
    "stream": false
  }' | \
  python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin), ensure_ascii=False, indent=2))'
```

返回标准 Chat Completions JSON，包含 `choices` 和 `usage`，回答位于 `choices[0].message`。`/v1/completions` 同样按该 `model` 路由，但请求使用 `prompt`，不是 `messages`。缺失 `model` 返回 400，未知模型返回 404。H2O 是决策模型，接口兼容不代表通用聊天能力；绕过 shim 不会自动执行其提示构造、概率校准及类型化答案转换。

**标准文本补全：** `/v1/completions` 使用 `prompt`，不使用 `messages`、`state`、`questions` 或 `attention_mode`。它仍直接到 vLLM，不执行 shim 校准；如需类型化答案和概率，优先使用上面的决策接口。

```bash
set -o pipefail
curl --fail-with-body --show-error --connect-timeout 5 --max-time 10 \
  'http://127.0.0.1/v1/models'

curl --fail-with-body --show-error --connect-timeout 5 --max-time 10 \
  --get 'http://127.0.0.1/v1/models' \
  --data-urlencode 'model=h2o-lightning-4b'

curl --fail-with-body --show-error --connect-timeout 5 --max-time 1800 \
  'http://127.0.0.1/v1/completions' \
  --header 'Content-Type: application/json' \
  --data-binary '{"model":"h2o-lightning-4b","prompt":"仅回复选项字母：结账页面对所有客户返回 HTTP 500，没有订单成功，应选 A) P1：营收路径完全不可用 还是 B) P3：仅外观问题？\n答案：","temperature":0,"max_tokens":64,"stream":false}' | \
  python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin), ensure_ascii=False, indent=2))'
```

成功需 HTTP 200、`model=h2o-lightning-4b`、有效 `usage` 和 `choices[0].text`；本例应选 `A`。标准聊天读取 `choices[0].message.content`，文本补全读取 `choices[0].text`，不能混用。所有示例均先完成 HF 下载与 `./start.sh h2o`；首次应用新版路由的重载步骤见模型目录 README。

**BOX：图像目标定位。** 使用 `/v1/systemone` 的 `type=box`，不是 `/v1/decisions` 的题型。先把 `IMAGE` 替换为包含待定位目标的实际 PNG 文件，先用小图；BOX 在同一 vLLM Metal 引擎执行因果生成，即使默认决策模式为 two-way。

```bash
set -o pipefail
curl --fail-with-body --show-error --connect-timeout 5 --max-time 10 \
  'http://127.0.0.1/v1/models'

curl --fail-with-body --show-error --connect-timeout 5 --max-time 10 \
  --get 'http://127.0.0.1/v1/models' \
  --data-urlencode 'model=h2o-lightning-4b'

IMAGE="/实际路径/图片.png"
IMAGE_BASE64="$(base64 < "$IMAGE" | tr -d '\n')"
printf '{"state":{"photo":"data:image/png;base64,%s"},"attention_mode":"causal","questions":{"target":{"type":"box","instructions":"定位 `photo` 中的咖啡杯。"}}}' "$IMAGE_BASE64" | \
  curl --fail-with-body --show-error --connect-timeout 5 --max-time 1800 \
    'http://127.0.0.1/v1/systemone' \
    --header 'Content-Type: application/json' \
    --data-binary @- | \
  python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin), ensure_ascii=False, indent=2))'
```

成功需 HTTP 200、正确模型和 `attention_mode=causal`；`answers.target` 包含 `type=box`、`box.left/top/right/bottom`、`confidence` 与 `present`。坐标是相对图像左上角的 `[0,1]` 比例，边界应有序，概率应有限且在 `[0,1]`；还须结合图片确认确实框住目标。全图框且置信度0不证明定位成功。返回 JSON，不输出图片文件；本示例尚未进行真实 H2O 图像质量验收。



### Higgs：文本转语音

```bash
curl --fail-with-body --show-error --connect-timeout 5 --max-time 10 \
  --get 'http://127.0.0.1/v1/models' \
  --data-urlencode 'model=higgs-audio-v3-tts-4b'

curl --fail-with-body --show-error --connect-timeout 5 --max-time 600 \
  'http://127.0.0.1/v1/audio/speech' \
  --header 'Content-Type: application/json' \
  --data-binary '{
    "model": "higgs-audio-v3-tts-4b",
    "input": "你好，欢迎使用语音合成。这是一段本机模型测试。",
    "response_format": "wav",
    "stream": false,
    "seed": 42,
    "max_new_tokens": 512
  }' \
  --dump-header /tmp/higgs-test.headers \
  --output /tmp/higgs-test.wav &&
afplay /tmp/higgs-test.wav
```

curl 成功后自动使用 macOS `afplay` 播放，失败则不播放。音频和响应头保留在 `/tmp/higgs-test.wav`、`/tmp/higgs-test.headers`，输出目标为 24 kHz 单声道。使用 `max_new_tokens`，不是旧参数 `max_frames`。

### YuE2：歌词生成歌曲

```bash
curl --fail-with-body --show-error --connect-timeout 5 --max-time 10 \
  --get 'http://127.0.0.1/v1/models' \
  --data-urlencode 'model=yue2-3b'

curl --fail-with-body --show-error --connect-timeout 5 --max-time 1800 \
  'http://127.0.0.1/v1/audio/speech' \
  --header 'Content-Type: application/json' \
  --data-binary '{
    "model": "yue2-3b",
    "input": "[Verse]\n今晚星光照亮街道\n我们一起追逐微笑",
    "instructions": "Mandarin pop, warm lead vocal, piano and acoustic guitar, at 100 BPM",
    "extra_params": {"cot": "off"},
    "response_format": "wav",
    "stream": false,
    "seed": 42,
    "max_new_tokens": 1024
  }' \
  --dump-header /tmp/yue2-test.headers \
  --output /tmp/yue2-test.wav &&
afplay /tmp/yue2-test.wav
```

`input` 是歌词，`instructions` 是音乐风格。curl 成功后自动播放，失败则不播放；保留 `/tmp/yue2-test.wav` 和 `/tmp/yue2-test.headers`，输出目标为 48 kHz 双声道。200 帧约 8 秒，只做短输出检查，可能截断歌词；完整歌曲应提高帧预算（支持 200–9000）并检查响应头中的截断标记。`cot=off` 不需要 ABC 计划，不提供参考音频或流式输出。

### MOSS：音频转写与说话人分离

将 `file` 路径替换为实际 WAV；下面沿用已测试的本机文件。

```bash
curl --fail-with-body --show-error --connect-timeout 5 --max-time 10 \
  --get 'http://127.0.0.1/v1/models' \
  --data-urlencode 'model=moss-transcribe-diarize'

curl --fail-with-body --show-error --connect-timeout 5 --max-time 1800 \
  'http://127.0.0.1/v1/audio/transcriptions' \
  --form 'model=moss-transcribe-diarize' \
  --form 'file=@/Users/zhaodehua/Projects/paggy-media/assets/voices/dabin_man.wav;type=audio/wav' \
  --form 'response_format=diarized_json' \
  --form 'temperature=0' \
  --form 'max_completion_tokens=2048'
```

返回 JSON 包含 `text`、音频 `duration` 和带 `start` / `end` / `speaker` 的 `segments`。该 WAV 的已测结果为 30 秒、5 段、同一说话人 `S01`；这不能证明多人分离准确率。整个 multipart 请求不超过 25 MiB，先用短音频测试；不要手动设置 multipart `Content-Type`，boundary 由 curl 生成。

### PaddleOCR-VL：单图文档识别

使用 `paddleocr-vl` 识别[官方示例图](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png)。`image_url.url` 是图片地址，`OCR:` 是官方识别任务提示；本机模型服务需要能访问该可信图片 URL。

初次接入先确认可以中断请求，再运行 `./start.sh gateway` 更新代理，随后 `./start.sh paddleocr`。主权重约 1.92 GB；共享后端升级可能停止其他模型，先确认下载与内存预算。架构兼容性、远程代码信任和本地图片用法见 [部署说明](paddleocr-vl/README.md)。

```bash
set -o pipefail
curl --fail-with-body --show-error --connect-timeout 5 --max-time 10 \
  --get 'http://127.0.0.1/v1/models' \
  --data-urlencode 'model=paddleocr-vl'

curl --fail-with-body --show-error --connect-timeout 5 --max-time 1800 \
  'http://127.0.0.1/v1/chat/completions' \
  --header 'Content-Type: application/json' \
  --data-binary '{
    "model": "paddleocr-vl",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {
              "url": "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png"
            }
          },
          {
            "type": "text",
            "text": "OCR:"
          }
        ]
      }
    ],
    "temperature": 0,
    "max_tokens": 2048,
    "stream": false
  }' | \
  python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin), ensure_ascii=False, indent=2))'
```

返回标准 Chat Completions JSON，识别文字位于 `choices[0].message.content`，中文解码与缩进展示方式同 H2O。成功需 HTTP 200、正确模型、非空文字、`finish_reason=stop`，并与示例图比对；`length` 表示达到输出上限，不算完整识别成功。此示例只在终端展示，不保存文件。

**尚未下载、启动或完成真实 OCR 验收**。表格/公式/图表可改用提示 `Table Recognition:` / `Formula Recognition:` / `Chart Recognition:`，这些任务及 SSE、取消、生产并发均未验证；单图 OCR 不等于完整 PDF 版面解析，本次不部署 PP-DocLayoutV2。

### Qwen-Image-2.1-Turbo：文本生成与图片编辑

使用官方 `Qwen/Qwen-Image-2.1-Turbo`，对外 ID 为 `qwen-image-2.1-turbo`，通过 Omni 原生 `QwenImage21Pipeline` 和 Omni Metal 执行。先确认磁盘/内存预算：权重约 **30.21 GiB**，不能承诺与其他大模型在 48 GiB Mac 并驻；勿为测试单模型启动 `all`。初次接入先确认可中断请求，再 `./start.sh gateway` 更新路由、`./start.sh qwenimage` 启动模型。共享后端升级可能停止其他模型，详见 [部署说明](qwen-image-2.1-turbo/README.md)。

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

沿用音频示例的媒体文件输出方式：curl 成功后才打开图片，失败不打开。PNG 和响应头保存在 `/tmp/qwen-image-test.png`、`/tmp/qwen-image-test.headers`，复测会覆盖；不能将错误 JSON 或旧文件当作图片验收。`response_format=file`、`n=1` 返回 PNG 二进制，不送入 JSON 格式化管道；默认 `b64_json` 则返回 `data[0].b64_json`。此接口用顶层 `prompt` / `size` / `n`，不是聊天接口的 `messages` / `extra_body`。

保留官方 8 步 `sample_sigmas`，`true_cfg_scale=1.0`；不是普通 Qwen-Image 的 50 步配置。2026-10-16 本次已修复通用 Metal diffusion SDPA 的 OOT 分派并重建共享非 editable wheel，`./start.sh qwenimage` 与聚合/单模型身份检查通过。统一入口上述 512×512 请求返回 HTTP 200、`image/png`、1,049,508 字节，耗时 27.70 秒；PNG 可解码为 512×512 RGBA，有效非单色，目视为窗台橘猫与蓝天。未验证 2K、编辑、多图、LoRA、seed 字节一致性、流式或生产并发。许可为 Qwen Research License，商用前核对模型卡 `LICENSE`。

#### 图片编辑：游艇草图转照片

先启动 `./start.sh qwenimage`，确认不带参数的 `/v1/models` 聚合列表包含规范 ID。下面对应 [官方草图](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/resolve/main/assets/turbo-boat-input.png)和给定 Diffusers 提示词，使用原生 `/v1/images/edits`，保留 2048×2048、8 步、CFG=1、CPU generator 和 seed `20261009`。**2K 编辑尚未真机推理验收**；48 GiB 机器须先确认内存余量，首次可将 `size` 改为 `512x512`，不要与其他大模型并驻。

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

此编辑端点当前只支持 `response_format=b64_json`，不能照搬文生图的 `file`；管道仅解码图片字段，不将 PNG 当 JSON 展示，HTTP 错误体会显示且返回非零，失败不打开旧图片。结果为 `/tmp/qwen-image-edit.png` 和 `/tmp/qwen-image-edit.headers`，复测覆盖；成功需 HTTP 200、JSON 中有可解码的 `data[0].b64_json`、PNG 尺寸匹配、RGBA 有效非单色，并对照草图检查船体结构、颜色与场景，不仅检查下载成功。

官方草图为 3,200×1,120 RGBA、1,878,998 字节，超过当前 **64 KiB 整个请求体**上限，故使用后端支持的 `url` 表单字段，不直接上传。HEAD 请求先解析 Hugging Face 重定向，后端再下载最终图片 URL；该 URL 可能过期，重试时重新执行整段，不保存或硬编码签名地址。客户端和后端均需能访问该站点；自行替换 URL 时只使用可信、可直接返回图片的 HTTPS 地址。不要手设 multipart Content-Type/boundary。本次已核对运行中 OpenAPI，并用实际后端下载器加载此 RGBA 草图，**未执行 2048×2048 编辑生成**。

`use_kv_cache=True` 是 Diffusers 参数，不是当前 HTTP 编辑接口字段；不要添加同名表单项冒充已生效。当前原生 pipeline 根据 transformer 的 `causal_condition` 自动建立每请求 KV cache，本地官方配置为 `true`，无需额外传参；不承诺与 Diffusers 输出逐字节一致。

#### QwenImage 的 GPU／ANE 优化边界

模型私有 `prepare_runtime.py` 从共享后端的官方 Omni commit 派生带 `+cpu.qwenmetal1` 标记的非 editable wheel，单独绑定通用 diffusion 算子，不覆盖模型 registry，也不改上游安装、模型采样或整模型 BF16。详细校准命令见 [模型 README](qwen-image-2.1-turbo/README.md#gpuane-算子接入与离线校准)。

- `qwen-image-2.1-turbo/start.sh` 默认将`QWEN_IMAGE_METAL_GPU=auto`、`VLLM_OMNI_METAL_ANE=off`写入launchctl plist，普通`./start.sh qwenimage`无需额外设置。显式环境变量仍可覆盖；GPU可用`QWEN_IMAGE_METAL_GPU=off`关闭。auto优先选择匹配AOT的SwiGLU和interleaved RoPE；缺产物时保留已有后备，不在启动或请求期自动编译。ANE默认关闭，不因GPU auto而开启。
- ANE 使用通用 `TiledDenseSwiGLU`；4096→12288→4096 沿中间维分为 10240 与 2048，保持每片不超过原 256 MiB 限制。只有显式 FP16、完整生成校准和匹配 profile 均满足时才协同；默认只绑定可配置的第 0 层，不代表全部 32 层已启用。
- 真实第 0 层权重、小张量输入已完成两片 CoreML/ANE preferred 图执行：相对 L2 0.297%、峰值归一误差 0.455%，保留 GPU 行；没有发布整模型 profile。这不是模型实际激活或整图质量／性能验收。
- 修复版overlay `6d492db20f717f02` 的真实A/B（512×512、seed42、8步）：off为27.325／27.228秒，auto为26.048／26.003秒，输出SHA一致、像素误差为0；候选每图实际命中512次RoPE AOT与256次SwiGLU AOT，RMSNorm仍走Torch。中位延迟降低约4.59%，每路仅两个正式样本，不外推其他尺寸或长期负载。部署默认现为GPU auto、ANE off，启动时仍执行共享版本预检。新版第0层ANE重测scaled max仍为12.94%，且中位耗时增加约2.29%，不发布profile。证据：相邻源码仓库`outputs/qwenimage-rope-aot-fixed-20261016-run1/`及`outputs/qwenimage-rope-aot-ane-20261016-run1/`。
- 当前启用验收：部署默认值及实际launchctl plist均为GPU auto、ANE off，运行修复版overlay `6d492db20f717f02`。总代理恢复后，统一聚合/单模型身份检查和真实512×512文生图通过：HTTP 200、image/png、512×512 RGBA、1,049,508字节，目视符合窗台橘猫提示。本次输出为`/tmp/qwen-image-gpu-aot.png`，响应头为`/tmp/qwen-image-gpu-aot.headers`；公开curl示例仍使用原`/tmp/qwen-image-test.*`文件名。

AR 分页 KV／paged-attention、NeoX 缓存表 RoPE、add-before-RMSNorm 不与 diffusion 的条件前缀 KV、interleaved RoPE、LayerNorm 等同，不直接套用；适配的是数学合同相符的路径，而不是为了同名“全部开启”改变模型语义。

### SeedVR2：7B 图片超分与视频修复

来源为 [Comfy-Org/SeedVR2](https://modelscope.cn/models/Comfy-Org/SeedVR2)，规范对外 ID **`seedvr2`**；固定普通 7B FP16，不是 3B、sharp 或量化版本。保留 Omni 原生 HTTP、调度、VAE 与采样，通过模型私有 Omni wheel 适配 7B 架构和内嵌 conditioning，不运行 ComfyUI。详见 [模型部署说明](seedvr2/README.md)。

**本次仅编写部署代码，不下载权重、不安装环境、不启动服务，真实权重加载、Metal 图片超分／视频修复质量与资源峰值尚未验收。** 以后实际部署前需确认约 15.82 GiB 权重、至少额外 5 GiB 下载空间及内存余量；不要为测试它启动 `all`。确认可以中断在途请求后，先 `./start.sh gateway` 更新路由，再 `./start.sh seedvr2`；共享后端升级可能停止其他模型。

启动后先请求不带参数的聚合列表，确认包含 `seedvr2`：

```bash
curl --fail-with-body --show-error --connect-timeout 5 --max-time 10 \
  'http://127.0.0.1/v1/models'
```

**图片超分：** 使用原生 `/v1/images/edits`，不是文生图接口。将 `/实际路径/input.png` 替换为存在的 PNG；例如 256×256 输入指定 `size=512x512` 为 2 倍超分。`size` 是输出宽×高，两个维度须为 16 的倍数；输入、输出每张均不得超过 407,040 像素，整个 multipart 请求不得超过 25 MiB。只接受单图、空格 prompt、单步 Euler 和 CFG=1，不支持文本编辑、mask、多图或透明通道保留。

```bash
set -o pipefail
curl --fail-with-body --show-error --connect-timeout 5 --max-time 10 \
  --get 'http://127.0.0.1/v1/models' \
  --data-urlencode 'model=seedvr2'

curl --fail-with-body --show-error --connect-timeout 5 --max-time 1800 \
  'http://127.0.0.1/v1/images/edits' \
  --form 'model=seedvr2' \
  --form 'prompt= ' \
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

该原生编辑接口只返回 JSON `b64_json`，不支持 `response_format=file`；管道仅将其中的 PNG 解码到 `/tmp/seedvr2-image-test.png`，不创建 JSON 结果文件。响应头保存在 `/tmp/seedvr2-image-test.headers`，其 Content-Type 应为 JSON；还需检查解码图片为可读、512×512、有效非空的 RGB PNG，并与输入对比细节、颜色与伪影。curl／解码失败不会打开旧文件。实际超分效果尚未验收。

**短视频修复：**

将 `/实际路径/short.mp4` 替换为存在的短 CFR 视频，首次建议 5 帧。整个 multipart 请求不得超过 25 MiB。`prompt` 必须是单个空格；非空文本不受支持。`size` 是明确的输出宽×高，两边须为 16 的倍数；先用 224×128 小尺寸核查执行路径，不据此承诺修复质量。

```bash
curl --fail-with-body --show-error --connect-timeout 5 --max-time 10 \
  --get 'http://127.0.0.1/v1/models' \
  --data-urlencode 'model=seedvr2'

curl --fail-with-body --show-error --connect-timeout 5 --max-time 1800 \
  'http://127.0.0.1/v1/videos/sync' \
  --form 'model=seedvr2' \
  --form 'prompt= ' \
  --form 'input_references=@/实际路径/short.mp4;type=video/mp4' \
  --form 'size=224x128' \
  --form 'num_inference_steps=1' \
  --form 'guidance_scale=1' \
  --form 'seed=7723' \
  --dump-header /tmp/seedvr2-test.headers \
  --output /tmp/seedvr2-test.mp4 &&
open /tmp/seedvr2-test.mp4
```

结果保存于 `/tmp/seedvr2-test.mp4` 与 `/tmp/seedvr2-test.headers`，复测覆盖；失败不打开文件，不把旧文件或 JSON 错误当成视频。成功需 HTTP 200、视频 Content-Type、可解码 MP4、输出尺寸与帧数正确、保留源帧率且有有效画面，并对比真实细节、颜色及时间连续性。省略 `fps` 保留源帧率；固定单步 Euler、CFG=1，不混用聊天接口。单 MPS 不支持需要至少四卡分片的 `/v1/seedvr2/restore-long`；未验证长视频、高分辨率、音轨保真、seed 一致性、取消或生产并发。

本次 SeedVR2 仅做离线代码验证：缓存 9 项、后端 11 项（含 tiny NaDiT CPU/MPS 对照）、路由边界 13 项通过，另通过命令替身启停和合成像素 PNG 输出合同冒烟；未构建私有运行时或启动任何服务，不将这些结果当成真实 7B 超分／视频修复验收。

### 限制与配置检查

SeedVR2 整个 multipart 请求 25 MiB / 1800 秒。Qwen-Image 整个请求体（包括文生图 JSON、编辑 multipart 表单）64 KiB / 1800 秒；URL 输入由后端另行下载，不能将此限制理解为可上传 64 KiB 以上的图片。PaddleOCR-VL 整个 JSON 请求体 25 MiB / 1800 秒。保留原限制：Higgs 16 KiB / 600 秒，YuE2 128 KiB / 1800 秒，MOSS 整个请求体 25 MiB / 1800 秒；H2O 标准接口和 Decisions 整个请求体 25 MiB，systemone 100 MiB / 1800 秒。为读取 model，非 systemone 请求体先缓存在内存（每请求最多 25 MiB），之后原样转发；不是流式上传。

部署配置检查：

```bash
bash -n start.sh stop.sh */start.sh */stop.sh
node --check router.mjs
docker compose config --quiet
docker compose -f h2o-lightning-4b/docker-compose.yml config --quiet
```

回归测试在编辑期间临时编写和执行，完成验证后按 `AGENTS.md` 的最后清理规则删除，不随部署文件保留。上文记录的是本次实际结果，不代表清理后仍存在相应测试入口；图片超分与视频修复 curl 保持可复制。

日志：`docker compose logs --tail 100 gateway`。配置修改通过构建镜像生效，不使用 Documents bind mount。

仅绑定本机，无鉴权；不要暴露公网，也不要把网关健康检查当成推理验收。
