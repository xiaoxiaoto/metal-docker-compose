# 本机模型总代理

统一入口：`http://127.0.0.1/v1/`（本机 80 端口）。模型部署目录统一位于 `composes/<模型目录>/`；根目录 Docker 只运行一个 Nginx + njs 代理，Metal 模型继续在 macOS launchctl 中独立运行。

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
| `wevisone-vl-4b` | WeVisOne-VL-4B vLLM + Metal，`8101`（7 个 `wevisone_task` 任务，含分割与抠图） |

`GET /v1/models` 不需要 `model` 参数，并行查询八个后端，只列出实际返回且与路由配置匹配的模型；后端不可达、超时或响应异常时跳过并记录日志，全部不可用时返回空列表。每个模型保留后端的 `id`、`object`、`created`、`owned_by`，不暴露本地模型路径。带 `?model=...` 的旧查询仍按模型转发，用于单后端身份检查。

```bash
curl -fsS http://127.0.0.1/v1/models
# {"object":"list","data":[{"id":"h2o-lightning-4b","object":"model","created":1791968752,"owned_by":"vllm"}, ...]}
```

`/v1/systemone` 经本机生命周期代理转发到 H2O shim `8740`，不检查请求里的 model。精确路径 `/v1/decisions` 则必须先通过 njs 原有的请求体类型、model 和大小校验，且只支持 `h2o-lightning-4b`，再由 `runtime_mgmt/model-lifecycle.json` 的固定 `path_upstreams` 转发到同一 shim；H2O 标准请求（包括 `/v1/chat/completions`、`/v1/completions`、带 model 的 `/v1/models`）仍转发到 vLLM `8000`，不经过 shim。除精确 `/v1/systemone` 外，`/v1/` 请求仍由 njs 从 JSON、multipart 普通表单字段、URL-encoded 表单或查询参数读取 `model`。模型进程、请求方法、原始路径、查询参数和请求体语义不改变；响应保留二进制音频、SSE 和后端状态码。

重复表单 model、查询/请求体 model 冲突和缺失 model 返回 400；未知模型返回 404；不支持的请求体类型返回 415；超出大小限制返回 413。通过上述校验后，已知非 H2O 模型请求 `/v1/decisions` 返回明确不支持的 400，不转发到其他模型后端；`/v1/decisions/` 返回 404，不接受末尾斜杠绕过。动态模型的推理请求等待冷启动；未准备、启动失败或超时返回 503。常驻但未启动的后端返回 502。

部署名称统一为模型卡仓库名 `/` 后的完整文本转为全小写，保留连字符等其余字符，不带作者前缀或 `:latest`。例如官方 `h2oai/h2o-lightning-4b` → `h2o-lightning-4b`；Higgs 使用实际模型卡 `bosonai/higgs-audio-v3-tts-4b` → `higgs-audio-v3-tts-4b`，不再使用手写别名。下载仓库名和缓存路径仍保留作者前缀；目录名、launchctl label 和启停简称不变。

模型名必须与后端 `/v1/models` 的 ID 完全一致。更换仓库时，同步更新 `router.mjs` 的 `targets` 和根目录 `start.sh` 的模型身份检查；不根据用户输入拼接上游地址。已有部署改名后需执行 `./stop.sh all && ./start.sh all` 重载模型和代理，会中断在途请求；调用方的 `model` 字段也须同步更新，旧名称返回 404。

## 基础 vLLM 与 Omni 的共享优化

基础 vLLM Metal 与 Omni Metal 使用相同的公共控制参数，但执行器、支持的 provider、
profile 合同和生命周期各自独立；参数名称统一不意味着所有能力均可跨执行器使用。
`METAL_MODEL_OPTIMIZERS` 默认空，未显式选择的优化保持关闭；量化、GPU 算子、
ANE 与 streaming 的职责边界及兼容／验证要求记录在 [`AGENTS.md`](AGENTS.md)。

扩展源码统一位于根目录 `extensions/`，按上游分为 `extensions/vllm/`、
`extensions/vllm_metal/`、`extensions/vllm_omni/` 和 `extensions/vllm_omni_metal/`。
其中 `extensions/vllm_metal/` 映射为发行包 `vllm_metal_extensions`，发行名
`vllm-metal-extensions` 仍为正确的 distribution name；功能源码按 `quantization/`、
`ops/`、`ane/`、`execution/` 分包，共享加载入口和生命周期代码留在包根。发行元数据、
依赖、子包、provider 和目录映射只由根 `pyproject.toml` 声明，依赖由根 `uv.lock`
锁定。公共环境必须使用 `uv sync --locked --no-dev --no-editable` 安装根包，不得使用
`--no-install-project`。`common_env/prepare_optimizations.py` 根据固定版本分别构建 Metal、Omni Metal
补丁 wheel。补丁按目标上游归属：vLLM 补丁位于 `extensions/vllm/`，vLLM Metal 补丁位于
`extensions/vllm_metal/patches/`，Omni / 模型专属补丁位于 `extensions/vllm_omni/` 对应
子目录，Omni Metal 补丁位于 `extensions/vllm_omni_metal/`；部署准备器只引用这些集中路径
并纳入指纹。H2O 私有 wheel 仍按共享 hook 后叠加业务双向补丁并继承公共扩展。
共享 Omni Metal 补丁保持 SHM 清理返回值合同：实际 unlink 返回 true，缺失返回 false；上游按前缀统计资源回收数量，不改变 codec 或模型拓扑。


## 通用 DynamicEngine 量化安装合同

公共量化参数 `METAL_QUANTIZATION=off|dynamic` 默认 off，与下节生命周期的
dynamic 模式无关。dynamic 必须在 `METAL_MODEL_OPTIMIZERS` 中显式选择
`dynamic_quantization`，并以 `METAL_QUANTIZATION_PROFILE` 提供绝对路径。
批准文件必须为当前用户持有的非符号链接普通文件、权限 0600、大小不超过 4 MiB，
且符合执行器支持的批准 schema；不得跨执行器直接复用 profile。
ANE 和 streaming 分别通过显式选择 `ane`、`streaming` 及绝对路径
`METAL_ANE_PROFILE`、`METAL_STREAMING_PROFILE` 启用。模型或框架私有优化
参数不再作为覆盖入口；不支持的执行器组合必须拒绝，不静默回退。

当前部署支持范围：H2O、MOSS、PaddleOCR 使用基础 Metal provider 合同；WeVisOne 的像素插件只接受无优化的 BF16 主模型，拒绝量化、ANE、streaming 和其他 provider。Higgs、
YuE2 和 SeedVR2 的 Omni 适配只接受 `dynamic_quantization`；Qwen-Image 接受
`dynamic_quantization` 和 `gpu_ops`。未实现的组合明确拒绝。
Higgs 与 YuE2 的旧 Omni ANE 后端开关在 plist 内固定关闭，不作为用户配置入口。
已运行的服务和已有 plist 不会因源码修改自动更新；重建共享后端会停止依赖模型并
可能中断在途请求，应先确认影响，再按对应模型 README 重新准备和验收。
Qwen-Image 与 SeedVR2 私有准备器固定 Omni `9534efe...`，根共享后端固定
`9842ebc...`，来源门禁拒绝该组合。需完成版本适配与验收，不得放宽门禁。
MOSS 私有准备器要求 vllm-metal 0.31.0，而共享锁使用 0.32.0；版本门禁会拒绝准备，须先适配并验证 STT。

共享及私有ready校验使用选定Python `-I` 导入
`quantization.capture`／`adapter`／`runtime`；共享指纹覆盖配置、锁文件及校验源码，Git commit 固定远端实现；私有 overlay 保留各自源码指纹。
四模型私有校准器由正常Omni wheel分发、校验源码SHA与RECORD归属，离线CLI仍从
各部署源码执行。普通nn.Linear、TP1未量化vLLM全投影、Diffusers未注册components
已接通用执行适配；普通MLX Module可用原生适配，不透明自管理对象须提供完整scope与apply hook，缺合同明确错误，
不静默转CPU/浮点。并非本表所有模型或任意Omni已移植。
FP16/BF16保持源dtype，FP32只identity；所有identity也合法，但不是量化收益。
实际候选和完整任务质量门禁见各模型 README；局部验证不能替代完整任务质量、
性能和峰值内存验证。

H2O 不再保留模型私有的 ANE／动态量化适配器和离线准备器。公共能力由根项目
`extensions/vllm_metal/` 提供并默认关闭；历史私有 profile 不迁移、不批准。基础包的
模块分层、兼容合同和验收规则见根 `AGENTS.md`。H2O 双向注意力、真正 FP32 输出头
和业务 shim 继续保留。规范模型 ID 仍为 `h2o-lightning-4b`，身份检查与真实推理 curl 不变。


## 通用模型生命周期

`runtime_mgmt/model_lifecycle.py` 是独立于 Metal、vLLM、Omni 和具体模型的标准库 HTTP 生命周期代理。每个服务只提供固定上游地址、argv 启停命令、就绪检查、模式和超时；不含模型名分支、量化或采样逻辑。`runtime_mgmt/lifecycle_launch.py` 是本部署的 macOS launchctl 适配器，不是通用核心的依赖。任意本机 HTTP 推理后端可使用相同配置合同接入。

部署配置为 `runtime_mgmt/model-lifecycle.json`：

| 模型 | mode | 空闲卸载 |
|---|---|---|
| `h2o-lightning-4b` | `resident` | 不自动卸载 |
| `higgs-audio-v3-tts-4b` | `dynamic` | 300 秒 |
| `yue2-3b` | `dynamic` | 300 秒 |
| `moss-transcribe-diarize` | `dynamic` | 300 秒 |
| `paddleocr-vl` | `dynamic` | 300 秒 |
| `qwen-image-2.1-turbo` | `dynamic` | 300 秒 |
| `seedvr2` | `dynamic` | 300 秒 |
| `wevisone-vl-4b` | `dynamic` | 300 秒 |

WeVisOne 显式配置600秒启动、1800秒请求预算；网关命名路由等待2400秒，以覆盖冷启动与完整推理。公共优化默认关闭，与生命周期 `dynamic` 无关。

每项服务可单独设 `mode`、正数 `idle_timeout`、`startup_timeout`（默认600秒）、`request_timeout`（加载完成后的完整请求预算，默认1800秒；Higgs设为600秒）。`upstream` 与可选 `path_upstreams` 必须为固定 loopback HTTP 地址；`start_command` / `stop_command` 为 argv 数组，不执行 shell，也不拼接请求输入。`ready_path` 默认 `/health`；本部署用 `/v1/models` 和 `ready_model` 核对身份。通用核心可直接运行：

```bash
python3 -I runtime_mgmt/model_lifecycle.py --config /absolute/path/services.json --port 18100
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
python3 -I runtime_mgmt/lifecycle_control.py status
# 首次实际推理使用下文对应模型的 curl；冷启动需在 --max-time 中另留最多600秒
```

控制器将通用核心、适配器和配置复制到 `~/Library/Application Support/model-lifecycle/`，以 `com.local.model-lifecycle` 运行，监听 `127.0.0.1:18100`；日志位于 `~/Library/Logs/model-lifecycle*.log`。启动前核对现有任务和端口；配置/代码变化时拒绝静默重启。应用配置需先确认无在途请求，再 `./stop.sh lifecycle`、`./start.sh gateway`。管理器停止会回收其持有且空闲的动态服务，外部服务不受影响；在途请求拒绝停止。

配置复制安装后才生效，运行中的管理器不会热读源码目录中的 JSON。应用配置前先确认并排空在途请求；停止管理器和重建网关会影响统一入口。

从已运行的显式部署切到动态时：先确认该模型无在途请求，用原 `./stop.sh <目标>` 停止，再经总代理请求；下一次由管理器启动才归其所有。不修改或自动停止当前服务。H2O保持常驻，shim仍由原Compose管理。

冷启动额外检查保守空闲内存与磁盘：普通模型至少16GiB、Qwen-Image至少36GiB、SeedVR2至少24GiB，plist所在盘至少2GiB。可用内存按系统 `memory_pressure -Q` 与物理内存估算，可能保守拒绝；不主动驱逐其他模型或提高MPS水位。此检查不等于生成峰值保证；单纯串行加载也不能保证多模型并驻可行。SeedVR2保留1800秒启动等待，未准备时不会因配置动态就自动下载。


## 启停

先启动 Docker Desktop，在项目根目录执行：

```bash
./start.sh          # 默认也先停模型、删历史环境并重建，再启动管理器和总代理
./start.sh gateway  # 同上；更新代理配置时使用
./start.sh h2o      # 重建共享环境 + 总代理 + H2O 模型与 shim
./start.sh higgs    # 重建共享环境 + 总代理 + 准备动态 Higgs
./start.sh yue2     # 重建共享环境 + 总代理 + 准备动态 YuE2
./start.sh moss     # 重建共享环境 + 总代理 + 准备动态 MOSS
./start.sh paddleocr # 重建共享环境 + 总代理 + 准备动态 PaddleOCR-VL
./start.sh qwenimage # 重建共享环境 + 总代理 + 准备动态 Qwen-Image
./start.sh seedvr2   # 重建共享环境并准备 SeedVR2；不加载权重
./start.sh wevisone  # 重建共享环境 + 总代理 + 准备动态 WeVisOne-VL-4B
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
./stop.sh wevisone  # 只停止 WeVisOne-VL-4B
./stop.sh lifecycle # 安全停止管理器；有在途请求则拒绝
./stop.sh all      # 停止总代理及全部模型
./stop.sh --help
```

Higgs、YuE2 和 MOSS 的启停脚本不操作总代理。H2O shim 由 `composes/h2o-lightning-4b/docker-compose.yml` 管理，绑定 `127.0.0.1:8740`；H2O 启停脚本同时管理模型和 shim。

每次根目录启动都会先停用所有依赖模型并删除历史共享/私有环境，随后只恢复指定目标；`all` 在同一次共享重建后按原顺序处理全部目标，任一步失败停止后续操作。resident 显式加载并在最后校验聚合列表；dynamic 每次调用子目录 `start.sh --prepare-only` 重建所需私有环境和 plist，不加载权重。仅执行 `gateway` / `lifecycle` 不恢复模型部署，旧 plist 可能引用已删除的私有环境；使用模型前应执行对应目标，或使用 `all` 一次准备全部。HTTP 冷启动、健康、列表和配置登记均不替代真实推理验收。

`all` 的模型顺序为 `h2o higgs yue2 moss paddleocr qwenimage seedvr2 wevisone`；停止 `all` 先关闭网关和生命周期管理器，再按相反顺序停止模型。WeVisOne 部署目录为 [`composes/wevisone-vl-4b/`](composes/wevisone-vl-4b/README.md)，本机端口8101，launchctl label为 `com.local.wevisone-vl-4b-metal`；plist位于 `~/Library/Application Support/wevisone-vl-4b/com.local.wevisone-vl-4b-metal.plist`。

### 每次根目录启动前重建共享环境

公共环境的构建和校验脚本统一位于 `common_env/`，根目录保留 `pyproject.toml`、`uv.lock` 和 `.python-version` 作为 uv 项目配置。根目录所有有效启动目标均调用 `common_env/prepare_backend.py`，不复用已有 `.ready` 环境；帮助、非法参数和前置检查失败不执行删除。直接调用准备器也执行同一重建合同。HTTP 动态冷启动不调用准备器。

后端准备器读取 macOS 系统 HTTP/HTTPS 代理并传给 Git、uv 和 pip；已有同名大小写代理环境变量（包括空值）优先，不硬编码端口、不关闭 TLS 校验。准备完成前统一入口可能不可用，应等待根目录命令成功后再执行身份检查和真实推理 curl。

共享后端的 Python 版本、直接依赖与来源统一声明在根目录 `pyproject.toml`，完整运行依赖由 `uv.lock` 锁定，`.python-version` 选择 3.12。需预先安装 uv 0.11.7+；引导准备/版本校验脚本的 `python3` 需 3.11+（标准库 `tomllib`），共享后端固定为 **uv 托管的 CPython 3.12**，匹配 Metal wheel 的 cp312 ABI，不选系统、Homebrew 或 Conda 解释器。准备器从 `requires-python` 读取版本范围，先安装/发现托管解释器，再检查锁文件是否与声明一致；这些步骤失败均不停止现有服务。uv 的托管 Python 不属于共享环境清理范围，不要在依赖环境使用期间执行 `uv python uninstall`。

四个核心包均从 GitHub 固定 commit 的源码安装，来源统一在 `pyproject.toml` 声明，由 `uv.lock` 锁定解析结果：vLLM 来自 [vllm-project/vllm](https://github.com/vllm-project/vllm) 的 `5df63a7203cdb9328d824a27b682e77c7b9928be`，使用该 commit 的官方源码归档并锁定 SHA256，避免抓取整个 Git 历史；Metal 来自 [vllm-project/vllm-metal](https://github.com/vllm-project/vllm-metal) 的 `6d6f2c37f8242f1ff1a336534adf6c2d2fe2936d`（均 v0.32.0）；Omni 来自官方仓库 `9842ebc17cf1d9c6b440ecb6a79039162f68a1dd`；Omni Metal 来自 [指定仓库](https://github.com/xiaoxiaoto/vllm-omni-metal) 的 `5774df7d720b3a8c6d32035729b42c31e480f346`。uv 隔离构建非 editable wheel，不使用这些包的官方预编译 wheel，也不读取相邻源码仓库或运行其安装脚本。vLLM 使用上游 CPU 构建目标并禁用预编译扩展复用，它是 Metal 插件底座，不代表模型 CPU fallback。模型私有补丁/overlay 继续由各模型准备器维护，不并入基础依赖。

根扩展的 Omni 依赖按 PEP 440 固定公共版本 `0.32.0.dev0`，允许同一公共版本的模型私有 local 标记；共享环境仍由 `uv.lock` 和固定 commit 选择 `0.32.0.dev0+cpu`。私有 overlay 必须独立校验来源、补丁、实际版本和 RECORD，不能仅凭公共版本认定兼容。私有环境复用已校验基础环境的 pip，仍执行完整 `pip check`。

源码构建还需完整 Xcode、macOS SDK 26.2+、可用 Metal/clang++ 编译器和 Rust/Cargo，准备器在停止服务前检查工具链。`vllm-metal` 的 wheel 打包不会自动编译原生扩展，因此安装后由 `common_env/prepare_native.py` 调用上游构建接口，准备 MLX/MPS launcher 和 Metal 库，并将真实产物的 SHA256/大小登记到安装包 RECORD，维持 H2O 私有 overlay 的完整性合同；登记前后均核对原 wheel 源码，不能通过重算记录掩盖源码变更。随后构建 Omni Metal 的 Torch-stream launcher；这些操作只发生在准备阶段，不在请求期间编译。

锁定范围仅为 macOS arm64。vLLM 的 CPU 元数据还包含不适用于本机的 Linux x86_64 Triton 直接 URL，uv 在平台裁剪前会尝试读取；`tool.uv.override-dependencies` 仅将这条非目标平台依赖转换为带 Linux x86_64 marker 的版本约束，避免无关下载。不修改安装包依赖元数据、不排除 Mac 运行依赖，仍执行完整 `pip check`；以后扩展平台时须重新验证该规则。

升级须显式修改 `pyproject.toml` 的版本/来源 commit，再执行 `uv lock --python 3.12 --managed-python` 并验证新组合；仅刷新锁文件不代表已部署。启动只执行锁文件一致性检查和 `uv sync --locked`，不改锁、不跟随 main 升级，也不使用根目录 `.venv` 作为模型后端。不要用普通 `uv sync` 代替部署入口，它不会执行安全停机、MPS 构建验收或发布 `current`。

1. 获取共享准备锁，联网查询官方 vLLM / Metal 最新稳定标签与 Omni main commit，保留兼容版本门禁。当前核心为 vLLM/Metal `0.32.0`、Omni/Omni Metal `0.32.0.dev0`；新核心超出配置的版本组合直接失败，不删除旧环境或混装。Omni main 前移只报告，不替换配置中固定的 commit；停止服务前检查 uv 托管 Python 和锁文件一致性。
2. 原子 drain 并停止生命周期管理器；有在途、加载或未确认完成的请求时拒绝。随后按逆序执行全部模型停止入口，包括 H2O shim；任一步失败不删除环境。
3. 核对 launchctl 任务退出、无残留进程引用环境后，仅删除 `~/.local/share/local-model-backend/runtime/<16位指纹>/`、`overlays/<模型>/<16位指纹>/` 和 `current` 链接。拒绝符号链接环境及异常所有权；其他资产、权重、下载缓存和日志保留。
4. 设置 `UV_PROJECT_ENVIRONMENT` 指向当前指纹的空目录，以 `uv sync --locked --no-dev --no-editable` 安装根扩展包与完整锁定依赖；强制 uv 托管解释器，核对新环境的基础解释器与预检选择一致。源码包由 uv 隔离构建非 editable wheel，不修改本地源仓库、Homebrew 或系统 Python。
5. 显式准备 Metal 原生扩展/库与 Omni Metal 的 Torch-stream attention launcher，再以选定 Python `-I` 执行 `common_env/check_versions.py`、`common_env/check_backend.py` 和 `pip check`，验证源码来源/commit、模块、资源、CPU 数值合同、MPS、非 editable 路径及 launcher。配置/锁文件/校验源码参与指纹，全部通过且指纹未变才写 `.ready` 与 `backend.json` 并原子激活 `current`。失败不恢复已删除环境；下次执行从空目录重建。


单目标命令只恢复该目标，其余模型保持停止；动态目标的私有环境和 plist 随准备重新生成，精度/量化参数按本次环境变量与脚本合同选择，不承诺保留旧 plist 的手工设置。需要全部部署时执行一次 `./start.sh all`，不要连续执行多个根目录启动命令并期待前一个模型仍在运行。


共享后端及模型私有适配运行时位于 `~/.local/share/local-model-backend/`：基础环境在 `runtime/<fingerprint>/`，私有覆盖环境在 `overlays/<model>/<fingerprint>/`，`current` 指向通过验证的共享基础环境。模型 plist、工作目录和日志位于各自部署目录对应的用户路径。

`all` 不再同时加载全部模型：当前只有H2O常驻，其余等待实际推理；已被动态请求加载的多个模型仍可能暂时并驻，受各自TTL和资源门禁约束。根目录 `stop.sh` 无参数只停止总代理，管理器继续空闲回收；`stop.sh lifecycle` 原子关闭请求入口后停止管理器，有在途请求则拒绝。`stop.sh all` 在管理器拒绝停止时不会继续杀模型。单模型停止先经管理器处理：自有动态模型在空闲时卸载并更新状态，有在途或未确认结束任务时拒绝；外部服务沿用原显式停止脚本，仍可能中断旁路请求。配置仍为dynamic时，下一次推理可重新加载。持续禁用服务应关闭入口并调整配置，而不是只停止worker。

## 接口调用示例

以下命令均使用统一入口 `http://127.0.0.1/v1/`，无需鉴权。H2O 需先手动下载官方模型，再在根目录启动 `./start.sh h2o`；其他模型启动对应目标以准备部署。resident 等待加载完成；dynamic 登记后保持冷状态，第一条推理会冷启动。冷状态身份查询可能返回 503，先用 `python3 -I runtime_mgmt/lifecycle_control.py status` 确认配置，首次推理完成后再查聚合与单模型身份。示例推理超时主要覆盖热推理；冷请求需额外加启动预算（通常 600 秒，SeedVR2 为 1800 秒），客户端超时后仍运行的任务不视为空闲。

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
python3 composes/h2o-lightning-4b/prepare.py --check-model
```

`composes/h2o-lightning-4b/model-release.json` 固定必需文件的官方大小和摘要，包含完整权重 SHA256；准备器流式读取，不反序列化或执行模型代码。缺少新版文件、旧缓存、混用不同 revision 或额外根目录权重会明确失败，且在共享后端升级前检查。HF 快照的同仓库 blobs 相对链接也会保留到只读容器挂载；详细下载、运行合同与未来升级命令见 [H2O 部署说明](composes/h2o-lightning-4b/README.md)。


**部署基于 vLLM Metal，支持 two-way、one-way 与 BOX 的原生补丁路径。** `extensions/vllm_metal/patches/h2o-lightning-4b-bidirectional.patch` 固定官方 Metal `v0.32.0` commit `6d6f2c37f8242f1ff1a336534adf6c2d2fe2936d`；准备器先 `git apply --check`，再构建非 editable wheel 到 H2O 私有 overlay，不修改基础安装、共享 `current`，不引入社区 facade 或其他推理框架。shim 默认双向，保留官方启动探针；显式 `causal` 切单向，`type=box` 开启同引擎因果图像定位生成。安全部署关闭整个引擎的 prefix cache/chunked prefill，整段预算40960；one-way 也不复用前缀缓存。模型 BF16 与真正 FP32 head 合同不降低，详细限制见模型 README。

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
  'http://127.0.0.1/v1/models'

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
  'http://127.0.0.1/v1/models'

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

将 `file` 路径替换为存在的实际 WAV，先使用短音频。

```bash
set -o pipefail
curl --fail-with-body --show-error --connect-timeout 5 --max-time 10 \
  'http://127.0.0.1/v1/models'

curl --fail-with-body --show-error --connect-timeout 5 --max-time 10 \
  --get 'http://127.0.0.1/v1/models' \
  --data-urlencode 'model=moss-transcribe-diarize'

curl --fail-with-body --show-error --connect-timeout 5 --max-time 1800 \
  'http://127.0.0.1/v1/audio/transcriptions' \
  --form 'model=moss-transcribe-diarize' \
  --form 'file=@/实际路径/音频.wav;type=audio/wav' \
  --form 'response_format=diarized_json' \
  --form 'temperature=0' \
  --form 'max_completion_tokens=2048' | \
  python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin), ensure_ascii=False, indent=2))'
```

返回 JSON 包含 `text`、音频 `duration` 和带 `start` / `end` / `speaker` 的 `segments`。单说话人音频不能证明多人分离准确率。整个 multipart 请求不超过 25 MiB；不要手动设置 multipart `Content-Type`，boundary 由 curl 生成。

### PaddleOCR-VL：单图文档识别

使用 `paddleocr-vl` 识别[官方示例图](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png)。`image_url.url` 是图片地址，`OCR:` 是官方识别任务提示；本机模型服务需要能访问该可信图片 URL。

初次接入先确认可以中断请求，再运行 `./start.sh gateway` 更新代理，随后 `./start.sh paddleocr`。主权重约 1.92 GB；共享后端升级可能停止其他模型，先确认下载与内存预算。架构兼容性、远程代码信任和本地图片用法见 [部署说明](composes/paddleocr-vl/README.md)。

```bash
set -o pipefail
curl --fail-with-body --show-error --connect-timeout 5 --max-time 10 \
  'http://127.0.0.1/v1/models'

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

表格、公式和图表可分别使用提示 `Table Recognition:`、`Formula Recognition:`、`Chart Recognition:`；单图 OCR 不等于完整 PDF 版面解析。

### Qwen-Image-2.1-Turbo：文本生成与图片编辑

使用官方 `Qwen/Qwen-Image-2.1-Turbo`，对外 ID 为 `qwen-image-2.1-turbo`，通过 Omni 原生 `QwenImage21Pipeline` 和 Omni Metal 执行。先确认磁盘/内存预算：权重约 **30.21 GiB**，不能承诺与其他大模型在 48 GiB Mac 并驻；勿为测试单模型启动 `all`。初次接入先确认可中断请求，再 `./start.sh gateway` 更新路由、`./start.sh qwenimage` 启动模型。共享后端升级可能停止其他模型，详见 [部署说明](composes/qwen-image-2.1-turbo/README.md)。

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

使用官方 8 步 `sample_sigmas` 和 `true_cfg_scale=1.0`；输出须检查 HTTP 状态、图片格式、尺寸与内容。许可为 Qwen Research License，商用前核对模型卡 `LICENSE`。

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

官方草图超过 **64 KiB 整个请求体**上限，故使用后端支持的 `url` 表单字段，不直接上传。HEAD 请求先解析 Hugging Face 重定向，后端再下载最终图片 URL；该 URL 可能过期，重试时重新执行整段，不保存或硬编码签名地址。客户端和后端均需访问该站点；自行替换 URL 时只使用可信、可直接返回图片的 HTTPS 地址。不要手设 multipart Content-Type/boundary。

`use_kv_cache=True` 是 Diffusers 参数，不是当前 HTTP 编辑接口字段；不要添加同名表单项冒充已生效。当前原生 pipeline 根据 transformer 的 `causal_condition` 自动建立每请求 KV cache，本地官方配置为 `true`，无需额外传参；不承诺与 Diffusers 输出逐字节一致。

#### QwenImage 的 GPU／ANE 优化边界

模型私有 `prepare_runtime.py` 从共享后端的官方 Omni commit 派生带 `+cpu.qwenmetal1` 标记的非 editable wheel，单独绑定通用 diffusion 算子，不覆盖模型 registry，也不改上游安装、模型采样或整模型 BF16。详细校准命令见 [模型 README](composes/qwen-image-2.1-turbo/README.md#gpuane-算子接入与离线校准)。

- 优化默认关闭；必须显式选择受该执行器支持的公共 provider，不能依赖模型私有
  GPU／ANE 默认值或历史参数。AOT 产物须事先准备，不能在请求期编译。
- 此 Omni 执行器尚未实现公共 ANE provider 或联合配置，显式选择时拒绝；完整任务校准与匹配 profile 不可由局部算子验证替代。

AR 分页 KV／paged-attention、NeoX 缓存表 RoPE、add-before-RMSNorm 不与 diffusion 的条件前缀 KV、interleaved RoPE、LayerNorm 等同，不直接套用；适配的是数学合同相符的路径，而不是为了同名“全部开启”改变模型语义。

### SeedVR2：7B 图片超分与视频修复

来源为 [Comfy-Org/SeedVR2](https://modelscope.cn/models/Comfy-Org/SeedVR2)，规范对外 ID **`seedvr2`**；固定普通 7B FP16，不是 3B、sharp 或量化版本。保留 Omni 原生 HTTP、调度、VAE 与采样，通过模型私有 Omni wheel 适配 7B 架构和内嵌 conditioning，不运行 ComfyUI。详见 [模型部署说明](composes/seedvr2/README.md)。

部署前需确认约 15.82 GiB 权重、至少额外 5 GiB 下载空间及内存余量；不要为测试它启动 `all`。确认可以中断在途请求后，按共享后端与模型准备流程操作。

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


### WeVisOne-VL-4B：统一 chat 接口（检测、定位、计数、OCR、分割与抠图）

来源为 ModelScope 固定发布 [`tencent-community/WeVisOne-VL-4B`](https://modelscope.cn/models/tencent-community/WeVisOne-VL-4B)（revision `855d2a73ac0b929f668e6ac4f5deccc667aba99b`，部署文件与上游 HF `tencent/WeVisOne-VL-4B` 逐字节一致），规范模型 ID **`wevisone-vl-4b`**；使用基础 vLLM + vLLM Metal 的 `Qwen3VLForConditionalGeneration` / `qwen3_vl` 原生 BF16 路径，不使用 Omni。主模型约9.66 GB，像素解码器另约38.5 MB；模型私有插件在同一 vLLM 服务内提取特征、以 Torch-MPS FP32 运行官方解码器并返回 PNG，不另建 shim、不重复加载主模型，无 CPU 推理回退。公共优化保持关闭。

七个任务统一走 `POST /v1/chat/completions`，用请求体顶层 `wevisone_task` 分流；不带该字段的请求原样透传，行为与此前一致（包括流式 SSE），响应不附加 `pixel`。

| `wevisone_task` | `predictions` | `artifacts`（`objects>0` 时） |
|---|---|---|
| `detection` | 逐框 | `overlay.png` |
| `referring` | 指代表达单框 | `overlay.png` |
| `pointing` | 逐点 | `overlay.png` |
| `counting` | 逐框计数 | `overlay.png` |
| `ocr` | 逐文本框，`label` 为文字 | `overlay.png` |
| `segmentation` | 逐框 | `overlay.png`、`masks/mask_NNN.png`、`alpha.png`、`cutout.png`、`segmentation.png` |
| `matting` | 逐框 | `overlay.png`、`masks/mask_NNN.png`、`alpha.png`、`cutout.png` |

请求只接受一组 `messages`：可选 system（缺省 `You are a helpful assistant.`）、恰一个 user；content 中恰一个 `image_url`（HTTPS 或 JPEG/PNG data URL，不接受 HTTP、重定向、本地路径、多图或视频）和一段非空文本（≤4096字符）。该文本是**查询词**而不是完整提示，服务端按固定源码 `9a6bcd8...` 的[官方模板](https://github.com/Tencent/WeVisOne-VL/blob/9a6bcd85404505df0d05cfb5d97512ccacf35dee/wevisone_vl/prompts.py)生成提示；`ocr` 使用固定提示，文本仍须非空但不参与推理。`max_tokens` / `max_completion_tokens` 为1…4096（默认2048），`stream` 必须为 false；其余采样参数透传，缺省为 `temperature=0`、`top_p=0.95`、`top_k=50`、`repetition_penalty=1.05`，`stop` 与 `skip_special_tokens` 由服务端固定。

响应是标准 `chat.completion`：`choices[0].message.content` 保留原始对象标记，`choices[0].message.pixel` 恒存在，包含 `task`、`image_size`、`objects`、`predictions`（原图像素坐标，保留1位小数；框为 `[x1,y1,x2,y2]`）和 `artifacts`。所有图像是不带 `data:` 前缀的 base64 PNG，键名固定白名单；产物无状态内嵌，不返回下载链接，也不在服务端落盘或缓存，已返回的数据不随模型卸载或进程重启失效；`objects=0`（明确拒识或未检出）时 `predictions` 与 `artifacts` 均为空。坐标越界、框无面积、`<OBJ>` 与掩码数量不一致或生成被截断都返回错误，不产生伪产物；对象最多16个。错误为纯文本：400（参数、图片获取、截断）、404（模型不符）、413（请求/图片 >25 MiB 或 >4,194,304像素）、415（非 JSON）、499（客户端断开）、500（特征或解码失败）。旧私有端点 `/v1/wevisone/predict` 已弃用，保留一个 minor 版本并在响应头标注 `Deprecation`。

先确认下载空间、内存余量及可中断在途请求，再在根目录运行 `./start.sh wevisone`；它一次重建共享后端、更新网关并准备动态部署，无需先单独运行 `gateway` 再重复重建。冷状态先查看 `python3 -I runtime_mgmt/lifecycle_control.py status`；模型列表不会唤醒模型，首次推理后再核对聚合及单模型身份。详情见[模型部署说明](composes/wevisone-vl-4b/README.md)。

#### 内嵌 save_result 后处理

先在执行 curl 的同一个 Bash 会话中定义一次下面的 `save_result`；客户端 `python3` 需已安装 Pillow，不创建额外脚本。它只校验响应结构并把服务端返回的真实 PNG 落盘，不再解析 `<ref>` 自己画图，也不加载模型：校验 `pixel` 的任务、尺寸、`objects`、白名单与期望文件集合，逐个检查 PNG 的格式、尺寸和通道，然后覆盖写入专用目录；`objects=0` 不产生图像。任何校验失败都非零退出，且不保留半套旧结果。

```bash
python3 -c 'from PIL import Image'
save_result() {
  python3 -c '
import base64, io, json, sys
from pathlib import Path
from PIL import Image

TASKS = ("detection", "referring", "pointing", "counting", "ocr", "segmentation", "matting")

def save_result(output_dir, payload):
    if payload.get("object") != "chat.completion" or payload.get("model") != "wevisone-vl-4b":
        raise ValueError("响应不是 wevisone-vl-4b 的 chat.completion")
    choice = payload["choices"][0]
    message = choice["message"]
    pixel = message.get("pixel")
    text = message.get("content")
    if choice.get("finish_reason") != "stop":
        raise ValueError("输出未正常结束；不保存为完整结果")
    if not isinstance(text, str) or not text.strip() or not isinstance(pixel, dict):
        raise ValueError("响应缺少文本或 pixel 字段")
    task, size = pixel.get("task"), pixel.get("image_size")
    objects, predictions = pixel.get("objects"), pixel.get("predictions")
    artifacts = pixel.get("artifacts")
    if (task not in TASKS or not isinstance(size, list) or len(size) != 2
            or any(type(value) is not int or value < 1 for value in size)
            or type(objects) is not int or not 0 <= objects <= 16
            or not isinstance(predictions, list) or len(predictions) != objects
            or not isinstance(artifacts, dict)):
        raise ValueError("pixel 字段结构无效")
    for prediction in predictions:
        coords = prediction.get("coords")
        if (not isinstance(prediction, dict) or prediction.get("type") not in ("box", "point")
                or not isinstance(prediction.get("label"), str) or not isinstance(coords, list)
                or len(coords) != (4 if prediction["type"] == "box" else 2)
                or any(type(value) not in (int, float) for value in coords)):
            raise ValueError("predictions 结构无效")
    expected = set()
    if objects:
        if task in ("segmentation", "matting"):
            expected = {"overlay.png", "alpha.png", "cutout.png"}
            expected.update(f"masks/mask_{index:03d}.png" for index in range(objects))
            if task == "segmentation":
                expected.add("segmentation.png")
        else:
            expected = {"overlay.png"}
    if set(artifacts) != expected:
        raise ValueError("artifacts 与任务/对象数不匹配")
    decoded = {}
    for name, encoded in artifacts.items():
        data = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            mode = "RGBA" if name == "cutout.png" else "RGB" if name == "overlay.png" else "L"
            if image.format != "PNG" or list(image.size) != size or image.mode != mode:
                raise ValueError("像素文件的格式、尺寸或通道无效：" + name)
        decoded[name] = data
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    managed = {"overlay.png", "alpha.png", "cutout.png", "segmentation.png"}
    managed.update(f"masks/mask_{index:03d}.png" for index in range(16))
    for name in managed:
        (output / name).unlink(missing_ok=True)
    for name, data in decoded.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    metadata = {"model": payload["model"], "response": text,
                "finish_reason": choice["finish_reason"], "usage": payload.get("usage"),
                "pixel": {key: value for key, value in pixel.items() if key != "artifacts"},
                "artifacts": {name: str(output / name) for name in sorted(decoded)}}
    (output / "response.txt").write_text(text + "\n", encoding="utf-8")
    (output / "result.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print("已保存：", output / "response.txt", output / "result.json")
    if not objects:
        print("未找到目标：predictions 为空，未写入图像")

save_result(sys.argv[1], json.load(sys.stdin))
' "$@"
}
```

每条推理 curl 都内嵌调用该函数，写入专用 `/tmp/wevisone-<任务>/` 目录，重复执行覆盖。`result.json` 的 `predictions` 是原图像素坐标，`artifacts` 记录落盘路径；展示时用文件路径替代 base64，避免终端刷屏。`objects=0` 表示明确拒识或未检出，不是分割成功；成功后可用 `open /tmp/wevisone-segmentation/overlay.png` 查看叠加图。

#### 身份检查与检测

以下使用[官方仓库图片](https://raw.githubusercontent.com/Tencent/WeVisOne-VL/9a6bcd85404505df0d05cfb5d97512ccacf35dee/demo/examples/images/boys.jpg)检测人物；服务与客户端都需能访问此可信 URL。冷状态先执行聚合查询确认网关，再发送检测请求唤醒模型；随后执行单模型身份检查，必要时再查一次聚合列表，两者均应包含 `wevisone-vl-4b`。查询词为 `person`，服务端按官方模板生成完整提示，`skip_special_tokens` 与 `stop` 由服务端固定。

```bash
set -o pipefail
curl --fail-with-body --show-error --connect-timeout 5 --max-time 10 \
  'http://127.0.0.1/v1/models' | \
  python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin), ensure_ascii=False, indent=2))'

curl --fail-with-body --show-error --connect-timeout 5 --max-time 2400 \
  'http://127.0.0.1/v1/chat/completions' \
  --header 'Content-Type: application/json' \
  --data-binary '{
    "model": "wevisone-vl-4b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://raw.githubusercontent.com/Tencent/WeVisOne-VL/9a6bcd85404505df0d05cfb5d97512ccacf35dee/demo/examples/images/boys.jpg"}},
        {"type": "text", "text": "person"}
      ]}
    ],
    "wevisone_task": "detection",
    "temperature": 0,
    "max_tokens": 2048,
    "stream": false
  }' | \
  save_result /tmp/wevisone-detection

curl --fail-with-body --show-error --connect-timeout 5 --max-time 10 \
  --get 'http://127.0.0.1/v1/models' \
  --data-urlencode 'model=wevisone-vl-4b' | \
  python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin), ensure_ascii=False, indent=2))'
```

核对每个框覆盖画面中的人物及数量，并打开 `/tmp/wevisone-detection/overlay.png` 对照原图；框正确不代表遮挡或远距离人物没有漏检。

#### 指代表达定位

查询词是完整指代表达，官方模板会补句末标点。检查返回框覆盖最左侧打鼓的男孩，而非整张图片。

```bash
set -o pipefail
curl --fail-with-body --show-error --connect-timeout 5 --max-time 2400 \
  'http://127.0.0.1/v1/chat/completions' \
  --header 'Content-Type: application/json' \
  --data-binary '{
    "model": "wevisone-vl-4b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://raw.githubusercontent.com/Tencent/WeVisOne-VL/9a6bcd85404505df0d05cfb5d97512ccacf35dee/demo/examples/images/boys.jpg"}},
        {"type": "text", "text": "the boy on the left"}
      ]}
    ],
    "wevisone_task": "referring",
    "temperature": 0,
    "max_tokens": 2048,
    "stream": false
  }' | \
  save_result /tmp/wevisone-referring
```

#### 指点

返回 `point` 类型预测，坐标不是边界框；检查点落在沙发靠枕上。

```bash
set -o pipefail
curl --fail-with-body --show-error --connect-timeout 5 --max-time 2400 \
  'http://127.0.0.1/v1/chat/completions' \
  --header 'Content-Type: application/json' \
  --data-binary '{
    "model": "wevisone-vl-4b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://raw.githubusercontent.com/Tencent/WeVisOne-VL/9a6bcd85404505df0d05cfb5d97512ccacf35dee/demo/examples/images/livingroom.jpg"}},
        {"type": "text", "text": "pillow"}
      ]}
    ],
    "wevisone_task": "pointing",
    "temperature": 0,
    "max_tokens": 2048,
    "stream": false
  }' | \
  save_result /tmp/wevisone-pointing
```

#### 计数

`counting` 使用官方框计数提示（`How many …? Please count them with boxes.`），`objects` 即总数；图片有四个靠枕，核对四个有效框，不把沙发坐垫计为靠枕。

```bash
set -o pipefail
curl --fail-with-body --show-error --connect-timeout 5 --max-time 2400 \
  'http://127.0.0.1/v1/chat/completions' \
  --header 'Content-Type: application/json' \
  --data-binary '{
    "model": "wevisone-vl-4b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://raw.githubusercontent.com/Tencent/WeVisOne-VL/9a6bcd85404505df0d05cfb5d97512ccacf35dee/demo/examples/images/livingroom.jpg"}},
        {"type": "text", "text": "pillows"}
      ]}
    ],
    "wevisone_task": "counting",
    "temperature": 0,
    "max_tokens": 2048,
    "stream": false
  }' | \
  save_result /tmp/wevisone-counting
```

#### OCR

`ocr` 使用官方固定提示 `detect all the word in the box format`，请求文本仍须非空但不参与推理；每个预测的 `label` 是识别出的文字。逐字对照招牌的 REAL COFFEE、Tea Café、10 METERS、AHEAD、TURN RIGHT、IN、SMALL ALLEY 等文字与定位，艺术字和重音符号可能识别有误。

```bash
set -o pipefail
curl --fail-with-body --show-error --connect-timeout 5 --max-time 2400 \
  'http://127.0.0.1/v1/chat/completions' \
  --header 'Content-Type: application/json' \
  --data-binary '{
    "model": "wevisone-vl-4b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://raw.githubusercontent.com/Tencent/WeVisOne-VL/9a6bcd85404505df0d05cfb5d97512ccacf35dee/demo/examples/images/coffee_sign.jpg"}},
        {"type": "text", "text": "all text"}
      ]}
    ],
    "wevisone_task": "ocr",
    "temperature": 0,
    "max_tokens": 2048,
    "stream": false
  }' | \
  save_result /tmp/wevisone-ocr
```

#### 分割与抠图

`segmentation` 与 `matting` 使用官方 alpha-matte 提示。服务端先由原生 chat 生成 `<OBJ>`，再在同一 Metal worker 中用已加载的 MLX BF16 主模型重算无 KV 缓存的特征，提取最终 RMSNorm 前的对象状态和四层 ViT merger 输入，官方像素解码器以 Torch-MPS FP32 执行；没有第二份 Transformers 主模型或 CPU 回退，框坐标不能代替真实掩码。像素解码串行执行，截断、特征不匹配或解码失败不返回伪结果。

本机图片需服务端可访问；原图不超过4,194,304像素、25 MiB，最多16个对象。

```bash
set -o pipefail
curl --fail-with-body --show-error --connect-timeout 5 --max-time 2400 \
  'http://127.0.0.1/v1/chat/completions' \
  --header 'Content-Type: application/json' \
  --data-binary '{
    "model": "wevisone-vl-4b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://raw.githubusercontent.com/Tencent/WeVisOne-VL/9a6bcd85404505df0d05cfb5d97512ccacf35dee/demo/examples/images/seabreeze.jpg"}},
        {"type": "text", "text": "person"}
      ]}
    ],
    "wevisone_task": "segmentation",
    "temperature": 0,
    "max_tokens": 2048,
    "stream": false
  }' | \
  save_result /tmp/wevisone-segmentation
```

输出在 `/tmp/wevisone-segmentation/`：`response.txt`、`result.json`、`masks/mask_000.png` 等逐对象连续掩码、`alpha.png`（各对象概率的逐像素最大值）、`cutout.png`（RGBA）、`overlay.png` 和 `segmentation.png`。最后一项以浮点概率 `>=0.5` 阈值化，只含0/255；掩码及 alpha 保留8位连续灰度，不能当成二值图。核对 HTTP 200、`objects>0`、尺寸与原图一致、有效前景／背景和人物实际轮廓；坐标正确或文件存在不能代替分割质量验收。

```bash
set -o pipefail
curl --fail-with-body --show-error --connect-timeout 5 --max-time 2400 \
  'http://127.0.0.1/v1/chat/completions' \
  --header 'Content-Type: application/json' \
  --data-binary '{
    "model": "wevisone-vl-4b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://raw.githubusercontent.com/Tencent/WeVisOne-VL/9a6bcd85404505df0d05cfb5d97512ccacf35dee/demo/examples/images/seabreeze.jpg"}},
        {"type": "text", "text": "person"}
      ]}
    ],
    "wevisone_task": "matting",
    "temperature": 0,
    "max_tokens": 2048,
    "stream": false
  }' | \
  save_result /tmp/wevisone-matting
```

抠图输出与分割相同的逐对象掩码、`alpha.png`、`cutout.png` 和连续透明度 `overlay.png`，但不生成二值 `segmentation.png`。核对 alpha 有连续过渡、前景／背景有效、`cutout.png` 的透明通道等于 alpha、RGB 与原图一致，并检查发丝和背景泄漏。8位 PNG 会量化概率，不能据此声称无损浮点输出或专业抠图精度。

#### 普通 chat 直通（无 `wevisone_task`）

不设置 `wevisone_task` 时请求原样交给模型，响应仍是普通 `chat.completion`（没有 `pixel`，不能套用上面的 `save_result`），流式 SSE 行为不变。文字定位可使用官方 `ocr_grounding` 提示：

```bash
set -o pipefail
curl --fail-with-body --show-error --connect-timeout 5 --max-time 2400 \
  'http://127.0.0.1/v1/chat/completions' \
  --header 'Content-Type: application/json' \
  --data-binary '{
    "model": "wevisone-vl-4b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://raw.githubusercontent.com/Tencent/WeVisOne-VL/9a6bcd85404505df0d05cfb5d97512ccacf35dee/demo/examples/images/coffee_sign.jpg"}},
        {"type": "text", "text": "Locate the following text in the image: TURN RIGHT"}
      ]}
    ],
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 50,
    "repetition_penalty": 1.05,
    "max_tokens": 2048,
    "skip_special_tokens": false,
    "stop": ["<|im_end|>"],
    "stream": false
  }' | \
  python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin), ensure_ascii=False, indent=2))'
```

参考框指定最左侧打鼓男孩，坐标已按该图片宽高归一化到0…1000；换图必须重新计算参考框，核对相似人物定位：

```bash
set -o pipefail
curl --fail-with-body --show-error --connect-timeout 5 --max-time 2400 \
  'http://127.0.0.1/v1/chat/completions' \
  --header 'Content-Type: application/json' \
  --data-binary '{
    "model": "wevisone-vl-4b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://raw.githubusercontent.com/Tencent/WeVisOne-VL/9a6bcd85404505df0d05cfb5d97512ccacf35dee/demo/examples/images/boys.jpg"}},
        {"type": "text", "text": "Locate all similar objects in the image based on the reference boxes: <ref>object 1</ref><box><46><234><203><721></box>"}
      ]}
    ],
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 50,
    "repetition_penalty": 1.05,
    "max_tokens": 2048,
    "skip_special_tokens": false,
    "stop": ["<|im_end|>"],
    "stream": false
  }' | \
  python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin), ensure_ascii=False, indent=2))'
```

需要点计数或其它自由提示时同样不带 `wevisone_task`；这类请求只返回文本，不生成 `predictions` 或叠加图。

#### 迁移说明

旧的 `POST /v1/wevisone/predict` 仍可用，但响应头带 `Deprecation: true` 与 `Link: </v1/chat/completions>; rel="successor-version"`，下个 minor 版本删除。迁移时把 `task`/`image_url`/`query` 改写为 `wevisone_task` 与 `messages`，`max_new_tokens` 改为 `max_tokens`，其余解码参数可省略；旧的扁平响应字段 `response`、`image_size`、`objects`、`artifacts` 分别对应 `choices[0].message.content` 与 `message.pixel` 的同名字段。

### 限制与配置检查
WeVisOne-VL-4B 整个 JSON 请求体（含图片data URL的base64）25 MiB / 1800 秒推理，冷启动另有600秒，网关等待2400秒；带 `wevisone_task` 时 `max_tokens` 上限4096、不支持流式，原图不超过4,194,304像素。SeedVR2 整个 multipart 请求 25 MiB / 1800 秒。Qwen-Image 整个请求体（包括文生图 JSON、编辑 multipart 表单）64 KiB / 1800 秒；URL 输入由后端另行下载，不能将此限制理解为可上传 64 KiB 以上的图片。PaddleOCR-VL 整个 JSON 请求体 25 MiB / 1800 秒。保留原限制：Higgs 16 KiB / 600 秒，YuE2 128 KiB / 1800 秒，MOSS 整个请求体 25 MiB / 1800 秒；H2O 标准接口和 Decisions 整个请求体 25 MiB，systemone 100 MiB / 1800 秒。为读取 model，非 systemone 请求体先缓存在内存（每请求最多 25 MiB），之后原样转发；不是流式上传。
WeVisOne-VL-4B 整个 JSON 请求体（含图片data URL的base64）25 MiB / 1800 秒推理，冷启动另有600秒，网关等待2400秒。SeedVR2 整个 multipart 请求 25 MiB / 1800 秒。Qwen-Image 整个请求体（包括文生图 JSON、编辑 multipart 表单）64 KiB / 1800 秒；URL 输入由后端另行下载，不能将此限制理解为可上传 64 KiB 以上的图片。PaddleOCR-VL 整个 JSON 请求体 25 MiB / 1800 秒。保留原限制：Higgs 16 KiB / 600 秒，YuE2 128 KiB / 1800 秒，MOSS 整个请求体 25 MiB / 1800 秒；H2O 标准接口和 Decisions 整个请求体 25 MiB，systemone 100 MiB / 1800 秒。为读取 model，非 systemone 请求体先缓存在内存（每请求最多 25 MiB），之后原样转发；不是流式上传。
像素端点额外限制原图不超过4,194,304像素、单图最多16个对象，仅允许 HTTPS 或 JPEG/PNG data URL；不支持 HTTP、重定向、本地路径或流式像素输出。

部署配置检查：

```bash
bash -n start.sh stop.sh */start.sh */stop.sh
node --check router.mjs
docker compose config --quiet
docker compose -f composes/h2o-lightning-4b/docker-compose.yml config --quiet
```


日志：`docker compose logs --tail 100 gateway`。配置修改通过构建镜像生效，不使用 Documents bind mount。

仅绑定本机，无鉴权；不要暴露公网，也不要把网关健康检查当成推理验收。
