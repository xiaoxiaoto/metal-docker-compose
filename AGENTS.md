# AGENTS.md

## 项目与工作范围

本项目是 Apple Silicon Mac 上的本机模型部署集合。根目录 Docker 只运行 Nginx + njs 总代理；模型由 macOS `launchctl` 独立管理，H2O 业务 shim 单独使用 Compose。统一入口为 `http://127.0.0.1/v1/`。

- 沟通、文档和交付说明使用简体中文，代码标识符沿用现有风格。
- 修改前阅读根目录 `README.md`、相关模型目录的 `README.md` 和实际脚本；以实际配置为准，不把历史验证结果当作本次结果。
- 选择最小完整改动，复用现有模式、标准库与已有依赖，不增加无需求的框架或抽象。
- 修改共享逻辑前搜索所有调用者，同步维护配置、启停入口、验证与文档；非平凡逻辑和缺陷修复必须编写并执行小型回归测试。部署目录中的测试脚本仅用于本次验证，交付前按下方规则清理，不作为部署文件保留。

## 新增模型部署：必须遵循已有部署规则

新增部署资源的目录结构、脚本写法、配置方式、生命周期和验收流程必须与已有同类部署一致。优先参考 `higgs-tts-3/` 的 Omni 部署；单阶段、模型私有运行时或业务 shim 分别参考对应现有目录，不另造一套部署体系。

### 部署模型命名规范（新增、改名和测试前必查）

- 对外模型 ID 必须取**实际模型卡仓库名 `/` 后的完整文本转全小写**，保留连字符等其余字符，不带作者前缀、`:latest` 或手写别名。例如 `forcewithme66/h2o-lightning-4b` → `h2o-lightning-4b`，`bosonai/higgs-audio-v3-tts-4b` → `higgs-audio-v3-tts-4b`。
- 下载仓库名和缓存路径保留作者前缀；模型 ID、目录名、launchctl label、启停简称是不同概念，不因模型 ID 调整而随意重命名已有目录、label 或简称。
- 后端 `--served-model-name`、后端 `/v1/models` 的 ID、`router.mjs` 的映射键、根目录 `start.sh` 的身份检查、README 和所有 curl 请求的 `model` 必须完全一致；模型私有适配使用等价的 served-name 配置。
- 更换仓库或模型 ID 时，搜索旧名称并同步全部相关配置、示例和测试；明确通知调用方更新 `model`，旧名会返回 404。需要重载时先说明在途请求会中断，获得确认后按 README 操作，不只改文档或网关映射。

### 目录与资源

- 每个模型使用独立目录，提供 `README.md`、`prepare.py`、可执行的 `start.sh` / `stop.sh`；需要 Omni 部署配置时提供 `metal-deploy.yaml`，确有模型私有适配需求时才增加 `prepare_runtime.py`、适配模块或补丁。
- Shell 沿用 Bash、`set -euo pipefail` 和切换至脚本目录的方式；检查依赖、平台、端口、服务状态，错误明确退出。
- 模型发现、选择、下载和校验放在 `prepare.py`：复用 ModelScope 缓存规则，校验仓库名、路径边界、模型架构与权重完整性，不为发现缓存而执行缓存代码或反序列化不可信 pickle。
- Metal 模型继续由 `launchctl` 管理，不新增纯 Nginx 模型 Compose 或在 Docker 中运行 Metal。plist/runtime 放在 `~/Library/Application Support/`，日志放在 `~/Library/Logs/`，沿用 `com.local.*` label 与独立空闲端口。
- 保留 `spawn`、本机绑定、离线加载、有限的启动等待及失败清理；只清理本次启动的服务，不终止占用端口的其他进程，不自动杀掉已注册但不健康的服务。
- Omni 配置从匹配上游版本的官方 profile 派生，注明来源/commit；保留原生 topology、connectors、processors、采样和 codec 语义，只调整已验证的 Metal 后端与资源参数。不得为启动成功删除不支持检查、静默 CPU fallback 或提高 MPS 内存水位。

### 共享后端与完整接入

- 使用根目录 `prepare_backend.py` 准备的共享隔离后端，通过 `VLLM_BACKEND_PYTHON` 复用；实际版本由 `check_versions.py` 校验，不创建另一套随意升级的基础环境。
- 保持官方联网版本预检、兼容版本组、源码指纹、非 editable wheel 构建、`pip check` 与验证后激活流程。新版本超出已验证组合时明确失败，不能逐包盲升；升级需同步 pins、依赖、校验和测试。
- 模型专用补丁或 STT 等适配保留在模型私有覆盖中，不污染共享基础环境，不修改 Homebrew、系统 Python 或其他项目环境。
- 新模型必须同步接入根目录 `start.sh` / `stop.sh` 的目标、帮助、身份检查和 `all` 顺序，以及 `prepare_backend.py` 的服务清单；启动顺序与反向停止顺序保持对应。
- 同步更新 `router.mjs` 的固定模型映射、`nginx.conf` 的上游/命名路由/模型列表子请求和必要的请求大小及超时限制，再更新根目录与模型目录文档、相关测试。不得根据用户输入拼接上游地址。
- 保持代理与模型生命周期独立、默认只操作 `gateway`。单模型不干扰其他模型；共享后端升级可能停止其他服务时必须明确告知。
- 启动最后请求不带参数的统一入口 `/v1/models`，检查本次指定模型全部存在；缺失、格式错误或请求失败必须非零退出。模型列表、加载成功或 `/health` 200 均不能代替真实推理验收。

## vllm-omni-metal 问题：拉取源码、通用修复、验证并提交

发现问题确实属于 `vllm-omni-metal` 时，必须在其源码仓库修复根因，不仅修改部署参数绕过故障。

1. 优先使用相邻 `../vllm-omni-metal`（共享后端构建所用源码），先阅读其 `AGENTS.md`、`README.md`、架构与测试文档，并检查 `git status`、remote、分支和已有改动。缺少源码时从已确认的官方仓库克隆至该位置；不要猜测来源或覆盖现有目录。
2. 拉取代码前先安全 `git fetch` 并确认目标分支/commit；仅在工作区与分支状态允许时快进更新。遇到未提交改动、分叉或冲突时保留现场并报告，不自动 reset、clean、覆盖、stash 或重写用户提交。
3. 保存最小复现、预期行为、实际错误及硬件/系统/依赖版本，定位共享 Metal 平台、运行时、执行合同、attention、设备能力或兼容层的根因。属于上游模型或模型私有适配的问题应在正确位置处理，不硬塞入 Metal 插件。
4. **修复必须是 Metal 通用的**：基于设备能力或通用接口合同，不硬编码模型名、checkpoint 路径、采样率、codec、部署拓扑或本机特例；不得新增模型白名单、覆盖上游 registry、全局伪装 CUDA，或通过放宽校验/降低精度掩盖错误。
5. **修复必须可验证**：补充能暴露原问题的回归测试，确认修复前失败、修复后通过；运行该仓库要求的相关测试、非集成测试、lint 与打包/依赖检查。涉及 MPS 执行、数值、采样、KV 或多阶段传输时，还须在 Apple Silicon 真机运行对应最小测试，检查有效输出、精度/seed 合同与资源清理，并以其他代表性模型或模型无关用例验证通用性。
6. 通过共享后端构建流程安装修复后的非 editable wheel，验证实际加载的是修复产物；对受影响部署执行真实推理验收。保留可复现命令、结果和必要日志，不用 mock、health 200 或合成信号冒充真实模型验收。
7. 本任务明确要求：**验证通过后在 `vllm-omni-metal` 仓库提交本次修复和回归测试**，提交说明包含根因、通用性与验证命令；只暂存本次相关文件，不夹带用户已有改动。默认不推送远端；若需要推送或提交 PR，先确认目标与权限。
8. 缺少硬件、依赖、权重、权限或资源预算时明确列出阻塞项，不能宣称已验证或已完成修复提交；不得提交未经验证的候选修复作为完成结果。

## curl 测试与示例规范（部署交付必须提醒）

新增或修改部署时，根目录和模型目录 README 都要提供可复制的模型身份检查与真实推理 curl，遵循已有示例，不只给启动命令。

- **格式保持一致**：新增或修改 README 中的 curl 文档必须沿用已有同类模型示例，不为单个模型另定格式。JSON 接口参考 H2O：同一 Bash 代码块内先检查模型身份，再发送请求，并使用统一的 `python3 -c` 管道展示中文 JSON；音频接口沿用已有文件输出与播放格式。除非用户明确要求，不另加编号步骤、JSON 结果文件或内嵌 Python 校验脚本；自动验收逻辑另行说明，不替代公开 curl 示例。
- **统一入口**：验收及公开示例使用 `http://127.0.0.1/v1/`，不把直连后端端口当作统一入口验收；后端直连仅用于定位问题，并明确标注。
- **先检查身份再推理**：启动后检查不带 `model` 的 `GET /v1/models` 聚合列表；测试单后端路由使用 `--get .../v1/models --data-urlencode 'model=<规范模型 ID>'`，确认返回包含预期 ID。不要用 `/health` 成功替代这两项或真实推理。
- **错误与超时**：curl 示例使用 `--fail-with-body --show-error --connect-timeout 5 --max-time <秒数>`。模型列表使用 10 秒；Higgs 推理 600 秒，H2O / YuE2 / MOSS 推理 1800 秒。新模型采用匹配代理配置的有限超时，不无限等待；保留 HTTP 错误内容和非零退出状态。
- **JSON 请求与展示**：使用 `--header 'Content-Type: application/json'` 和 `--data-binary`。展示 JSON 时使用下方解码格式，中文不输出为 `\uXXXX`，不修改响应字段/数值；管道前设置 `set -o pipefail`，避免格式化成功掩盖 curl 失败。自动验收另需断言关键字段，缩进展示不等于内容正确。
- **音频上传**：转写使用 `--form 'model=...' --form 'file=@<实际文件>;type=audio/wav'`，不要手设 multipart `Content-Type` 或 boundary；明确要求替换为存在的实际音频，先用短音频。MOSS 的整个 multipart 请求不得超过 25 MiB。
- **音频输出**：使用 `--dump-header /tmp/<模型>-test.headers --output /tmp/<模型>-test.wav`，并通过 `&& afplay /tmp/<模型>-test.wav` 仅在 curl 成功后播放；不得把二进制音频送入 JSON 格式化管道。复测避免误用旧文件，检查 HTTP 状态、Content-Type、可解码性、非空/有效信号、采样率、声道及截断标记，不能把 JSON 错误体当 WAV。
- **参数遵循真实接口**：H2O `/v1/systemone` 无需 `model`，使用 `state` / `questions`；标准 `/v1/chat/completions` 使用 `model` / `messages`，`/v1/completions` 使用 `model` / `prompt`，不混用或靠末尾斜杠绕过路由。Higgs / YuE2 使用 `max_new_tokens`，不是旧参数 `max_frames`。
- **模型专项验收**：Higgs 目标为 24 kHz 单声道；YuE2 为 48 kHz 双声道，`input` 是歌词、`instructions` 是风格，短输出可能截断，不能宣称完整歌曲通过；MOSS 使用 `response_format=diarized_json`，检查 `text`、`duration`、`segments` 的时间与 speaker，单说话人测试不能证明多人分离准确率。
- **边界与流式测试**：改动路由时覆盖缺失/冲突/重复 model（400）、未知 model（404）、不支持的请求体类型（415）、超限（413）；验证二进制与后端状态码不被改写。涉及 SSE 时使用 `curl --no-buffer` 并检查实际事件流，不按单个 JSON 文档解析。

JSON 展示格式（按实际接口替换请求体）：

```bash
set -o pipefail
curl --fail-with-body --show-error --connect-timeout 5 --max-time 1800 \
  'http://127.0.0.1/v1/chat/completions' \
  --header 'Content-Type: application/json' \
  --data-binary '{"model":"h2o-lightning-4b","messages":[{"role":"user","content":"仅回复选项字母：营收路径完全不可用，应选 A) P1 还是 B) P3？"}],"temperature":0,"max_tokens":64,"stream":false}' | \
  python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin), ensure_ascii=False, indent=2))'
```

交付时提醒用户：使用哪个规范模型 ID、先启动哪个目标、执行哪些 curl、输出文件保存在哪里、如何判断成功，以及哪些能力尚未验证。只记录本次实际结果，不引用 README 的历史输出作为本次验收结论。

## 检查与验收

从项目根目录执行适用检查：

```bash
bash -n start.sh stop.sh */start.sh */stop.sh
node --check router.mjs
docker compose config --quiet
docker compose -f h2o-lightning-4b/docker-compose.yml config --quiet
```

修改 Python 时用 `ast.parse` 或在临时缓存位置进行编译检查，并执行相关回归测试。临时测试优先放在部署目录外；如需在部署目录内验证，完成后必须删除测试脚本、夹具与生成产物，并移除 README 中指向已删除测试文件的执行命令。不得把未执行的测试报告为通过；清理后的部署目录不要求保留测试入口。

修改部署时，按 README 使用 `./start.sh <目标>` 验证统一模型列表，再执行该模型真实推理请求；检查响应状态、输出格式及内容（音频还需检查采样率、声道和有效信号）。停止、重载会中断请求；大模型下载、真机长任务与多模型并驻前确认资源和影响，避免为验证单模型而启动 `all`。

## 每次编辑完成后的最后清理

- **每次编辑任务完成时，先完成修改、文档同步和适用验证，最后一步必须清理本部署目录中与部署无关的文件，再交付。** 不得在清理后继续修改文件或运行会重新生成缓存/产物的命令；如确需继续编辑或验证，结束后必须再次执行末尾清理。清理后仅做只读核对。
- **测试脚本不属于部署交付文件，必须删除**：包括 `test_*.py`、`*_test.py`、`*.test.mjs`、测试 Shell 脚本，以及仅供测试使用的夹具、替身、临时验证脚本、报告和生成媒体。不得因“以后可能复用”保留这些文件。
- 同时删除部署目录内无部署用途的检查缓存和编译产物，如 `__pycache__/`、`*.pyc`、`.pytest_cache/`、`.ruff_cache/`，以及本次创建的下载/构建/调试临时产物。根据实际用途判断，不凭文件名误删部署依赖。
- 保留部署必需的源码、配置、补丁、准备/安装/版本与模型校验脚本、启停入口、README 和本文件。被启动流程调用的 `check_*.py` 等校验程序属于部署依赖，不按测试脚本删除；公开图片与视频 curl 示例也必须保留。
- 清理前同步 README，移除指向待删除文件的命令；可以保留本次实际执行的验证结果与未验证边界，但不能将已删除的临时测试描述为仍可直接运行。
- 清理范围仅限本部署目录及本次明确创建的临时文件，不删除用户权重、模型缓存、虚拟环境、运行日志或其他项目文件。相邻 `vllm-omni-metal` 等源码仓库的回归测试不在此清理范围，仍遵循其修复、验证与提交要求。

## 安全与交付

- 只绑定本机且当前无鉴权，不暴露公网；保留请求边界、大小/超时限制与原始请求/响应转发语义，包括二进制音频、SSE 和后端状态码。
- 不输出或提交凭据、权重、虚拟环境、模型缓存、运行日志或生成媒体；不删除用户缓存、环境或已有改动。
- 当前部署目录不是 Git 仓库，不擅自 `git init`；上述修复提交只针对已确认的 `vllm-omni-metal` 源码仓库。
- 交付说明列出修改文件、实际检查、未验证项和已知风险；仅文档修改无需启动服务或下载模型。
