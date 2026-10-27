# H2O Lightning：本机 vLLM Metal + 官方 shim

来源：[H2O.ai 官方模型卡](https://huggingface.co/h2oai/h2o-lightning-4b)。本部署同步 **v1.3.1 权重**与官方 main tooling，统一固定到 commit [`e7b32490c9d2a994bd75a4d1ea7e45716080175e`](https://huggingface.co/h2oai/h2o-lightning-4b/tree/e7b32490c9d2a994bd75a4d1ea7e45716080175e)。对外模型 ID 保持 `h2o-lightning-4b`，端口和 launchctl label 不变。

部署目录现为 `h2o-lightning-4b/`。此次仅迁移项目路径；启停简称仍为 `h2o`，launchctl label、`~/Library/Application Support/h2o-lightning-shim/` 运行时目录和日志路径不变。Compose 显式保留原项目名与容器名 `h2o-lightning-shim`，确保改名后仍能管理已有容器，不产生另一套部署。

模型由用户通过 Hugging Face 手动下载，启动器不自动下载权重；旧魔搭镜像缓存保留但不再自动选择。当前固定发布已完成文件校验，正在按用户要求实际启动与验收；具体结果见末尾验证范围。

## 手动下载官方发布

使用已有 Hugging Face CLI（`hf`）；不要把新的依赖装进共享后端或系统 Python。以下命令仅供稍后手动执行，主权重约 **9.32 GB**，需预留下载空间。固定同一 commit 的权重、配置和 shim，不要只下载 `model.safetensors`，也不要把 `v1.3.1` tag 的旧 tooling 与 main 混用：

```bash
hf download h2oai/h2o-lightning-4b \
  --revision e7b32490c9d2a994bd75a4d1ea7e45716080175e \
  model.safetensors config.json generation_config.json tokenizer.json tokenizer_config.json \
  chat_template.jinja merges.txt vocab.json preprocessor_config.json video_preprocessor_config.json \
  serve_config.json h2o_lightning_shim.py assets.py box.py pooled_readout.py example_receipt.png
```

默认使用 HF 标准缓存 `~/.cache/huggingface/hub/models--h2oai--h2o-lightning-4b/snapshots/<上述 commit>/`；支持 `HF_HUB_CACHE`、`HF_HOME`、`XDG_CACHE_HOME`。下载和启动时保持相同环境变量。若用 `hf download ... --local-dir /实际目录`，启动时 `export H2O_MODEL_DIR=/实际目录`，覆盖自动缓存位置。

可在不启动服务、不联网的情况下先检查：

```bash
python3 h2o-lightning-4b/prepare.py --check-model
```

`model-release.json` 记录官方 API 提供的文件大小、Git blob ID 与 LFS SHA256。准备器流式核对全部必需文件（包括完整权重），不执行缓存里的代码、不反序列化 pickle，不自动下载或覆盖文件。HF 快照允许链接到同仓库 `blobs/`，其他外链拒绝；Compose 只读挂载该仓库根目录，保留快照相对链接，避免容器内断链。普通 `--local-dir` 目录直接只读挂载。缺文件、LFS 指针、摘要不符、旧 shim 或混入旧 `label_rows.safetensors` 等额外权重均明确失败。

## 运行合同与启停

需要 Apple Silicon、macOS ≥ 15、系统 Python ≥ 3.9、Docker Desktop 和根目录准备的 Python 3.12 共享后端。规范 ID 为 **`h2o-lightning-4b`**，原生 vLLM Metal 本机端口 `8000`、官方 shim `127.0.0.1:8740`、统一入口 `http://127.0.0.1/v1/`。

- **同一 vLLM Metal 引擎支持 two-way、one-way 与 BOX 三条路径。** shim 默认 `bidirectional`；请求 `attention_mode=bidirectional` / `two-way` 使用全注意力层双向读出，`causal` / `one-way` 使用单向读出，GDN 始终因果。BOX 是 `type=box` 的图像定位题，不是第三种 attention；它使用同一引擎的因果多 token 生成，再由官方 shim 解析坐标、校准置信度和读取目标存在概率。
- `patches/vllm-metal-bidirectional.patch` 基于官方 `vllm-metal v0.32.0` commit `6d6f2c37f8242f1ff1a336534adf6c2d2fe2936d`，不引入社区 MLX facade、其他 server 或 CUDA/Triton fork。Metal 本身使用的计算库沿用共享依赖，不创建另一套推理框架。
- 为保证双向语义，显式启用 `--additional-config '{\"bidirectional_attention\":true}'`，整个引擎关闭 prefix caching 与 chunked prefill，使用 `mamba-cache-mode=none`，40960 上下文与同等整段 batch token 预算。**取舍是 one-way 也不复用前缀缓存**；不声称实现官方 CUDA fork 的请求级缓存优化。双向仅允许单 token 读出，不用于自回归生成；量化 KV、speculative、PP/TP/DP、KV transfer 等未支持组合明确拒绝，不静默回退。
- 保留官方配置：题型温度 `choice=0.75`、`noul=0.8`、`score=0.65`，真假概率 floor `0.801`，图像温度 `0.65`；模型权重和 embedding lookup 保持 BF16，补丁按 `head_dtype=float32` 实施真正 FP32 输出投影，不是 BF16 计算后转型，不放宽官方 three-probe 容差。
- BOX 通过 `LIGHTNING_BOX_ENABLED=1` 开启，保留官方 `generate`、`generation` 置信度校准与 `presence` 设置；Compose 镜像增加固定 Pillow `13.0.0`，以支持官方绘框路径。保留每次最多4张图、每张1048576像素、`max-logprobs=128`、4并发和 `gpu-memory-utilization=0.55`。这些不是官方 H100 性能或基准结果的复现。
- 根目录 `./start.sh h2o` 每次离线校验权重后重建共享环境：安全停止生命周期管理器及全部依赖模型，删除历史共享环境和私有 overlay，再安装验证共享后端，随后构建 H2O 私有补丁 wheel 并加载。停止失败不删除；构建失败不回退旧环境。权重、下载缓存、量化资产和日志保留。缺权重时不重建、不启服务；详见[根目录重建合同](../README.md#每次根目录启动前重建共享环境)。
- 共享后端准备阶段会继承 macOS 系统 HTTP/HTTPS 代理，显式环境变量优先，保留 TLS 校验；无需在代码里填写代理端口。代理程序须已运行。首次后端安装完成前网关可能尚未监听，应等待 `./start.sh h2o` 最终成功，再执行下方身份检查与真实推理 curl。
- vLLM 由 `com.local.h2o-lightning-vllm` 管理，plist 位于 `~/Library/Application Support/h2o-lightning-shim/`，日志为 `~/Library/Logs/h2o-lightning-vllm.log` 和 `.error.log`。保持离线加载、`spawn`、有限等待和失败时只清理本次启动服务。已注册但不健康、配置不匹配或端口被其他进程占用时拒绝接管，不自动杀进程或覆盖运行配置。
- H2O 是根目录 `model-lifecycle.json` 中的 `resident`：不自动加载或空闲卸载。`/v1/systemone` 与精确 `/v1/decisions` 经生命周期代理转到 shim；后者必须通过 model 校验。`/v1/chat/completions`、`/v1/completions`、单模型 `/v1/models` 仍到 vLLM，不经过 shim 的提示构造、概率校准与类型转换。

### 补丁式安装边界

`prepare_runtime.py` 从固定官方 commit 导出干净副本（不覆盖相邻源码工作区），先 `git apply --check`，再应用补丁；补丁只添加请求级 attention mode 合同/调度与原生 FP32 head，不改 Metal shader。复用已安装官方发行 wheel 中经 RECORD 校验的同 ABI 原生扩展与 `.metallib`；原生源码必须逐字匹配，不能把旧二进制配给变化的 shader。Python 改动通过非 editable wheel 安装到 `~/.local/share/local-model-backend/overlays/h2o-lightning-4b/<fingerprint>/`，其余依赖通过现有 `.pth` 模式继承共享基础环境。

wheel 构建后逐文件比对归档与固定源码副本（包含 native resources），安装后校验私有模块路径、源码摘要、RECORD 与原生扩展加载；私有 `bin/vllm` launcher 指向该 overlay 的 Python。每次复用都重新检查版本、`pip check` 和加载，只有全部通过才发布 `.ready`。指纹覆盖固定基线、补丁、准备器、基础发行资源和解释器；不会修改已安装基础包、Homebrew、共享 `current` 或其他模型的 overlay；请求期间不下载、不打补丁、不编译。缺补丁、基线不匹配、依赖冲突或资源损坏均明确失败。

根目录启动现在统一负责安全排空、停用依赖模型、删除历史环境并重建，不需要手动先停 H2O。重建期间统一入口不可用，构建失败不会回退旧环境；在根目录执行：

```bash
./start.sh h2o
```

停止 lifecycle 会影响统一入口，有在途请求时拒绝停止。根目录每次启动（包括默认/gateway/lifecycle）都会安全停止所有依赖模型并删除历史环境重建；单目标只恢复 H2O，其余模型不自动恢复。不要连续执行多个根目录启动命令并期待前一个模型仍运行。

**验证边界：** 私有 wheel/RECORD、模块归属和原生 Metal 扩展加载检查不能替代真实推理。FP32 head 使用模型原生 forward 验证实际投影，不要求模型支持缺少原生 cache/位置初始化的独立 backbone 调用；不改变 selective logits 的能力门禁，不放宽官方 three-probe。实际启动与统一入口验收结果记录在末尾。

## 统一入口验收

以下示例与根目录 README 保持一致。先完成 HF 手动下载与上节启动/重载，再执行统一入口 curl；每组先查聚合列表和单模型身份，不用 `/health` 代替模型与真实推理验收。推理 JSON 直接显示在终端，不保存结果文件。

**Two-way：SystemOne choice / noul / score。** `/v1/systemone` 无需 `model`，使用 `state` 和对象形式的 `questions`；下例显式选择双向，一次询问优先级、是否影响所有客户与紧急程度。改成 `causal` 即可对同一请求比较 one-way；省略该字段默认双向。

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

成功需 HTTP 200、`model=h2o-lightning-4b`、`attention_mode=bidirectional`、`probability_source=native` 与有效 `usage`；官方启动探针必须通过，失败返回503，不能绕过。`answers` 是以问题名为键的对象：`answers.priority.choice` 应为 `p1`，`answers.all_users.noul` 是 `[0,1]` 内的真假概率，`answers.urgency.score` 是 `[0,2]` 内的等级期望值。choice 与 score 的概率分布应非负且归一化；缩进展示不替代字段与数值检查。

**One-way：官方 OpenAI Decisions predicate / choice / score。** 精确 `/v1/decisions` 必须提供 `model=h2o-lightning-4b`，使用 `input` 与数组形式 `questions`，不是 SystemOne 的 `state` / 对象形式 `questions`。三种题型分别使用真假命题、`choices` 和 `levels`；下例显式指定 `attention_mode=causal`，也可改成 `bidirectional`。

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

成功需 HTTP 200、模型身份匹配和有效 `usage`。Decisions 的 **`answers` 是数组**，按每项的 `name` 对应问题：`outage` 的 `probability` 在 `[0,1]` 内，`team.choice` 应为 `payments`，`urgency.score` 在 `[0,2]` 内；choice/score 的 `probabilities` 也是有效分布数组。这里的 `team.choice` 表示 `name=team` 那项的字段，并非对象路径。此接口不返回 SystemOne 的顶层 `attention_mode` 或 `probability_source`，也不使用 Chat Completions 的 `choices`。缺失/冲突/重复 model 返回 400，未知 model 返回 404，不支持请求体类型返回 415，模型请求超限返回 413；其他模型不支持此 Decisions 路由，尾斜杠不能绕过。

**标准聊天：** `/v1/chat/completions` 使用 `model` / `messages`，直接转到 vLLM，不执行 shim 的提示构造、概率校准或类型化转换，不传 `attention_mode`。

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

成功需 HTTP 200、`model=h2o-lightning-4b`、有效 `usage` 与 `choices[0].message.content`；本例应选 `A`。标准接口兼容不代表通用聊天能力；需要概率与类型化答案时使用上面的决策接口。

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


## 本次验证范围

2026-10-23：准备器成功构建官方 Metal `0.32.0` 的 H2O 私有 non-editable wheel；`pip check`、安装源码/native 资源摘要与 RECORD 校验、原生 `get_ops()` 导入校验通过。实际加载的 `vllm` core 位于共享基础环境，`vllm_metal` 与新增补丁模块来自 H2O overlay；overlay 的 `vllm --help` 和 `vllm serve --help` 均可运行。隔离烟测通过 attention mode 别名/非法值、未启用 bidirectional 拒绝、单 token 双向读出限制、全段配置合同，以及真实 MLX BF16 tied embedding 到 FP32 vocab 投影数值一致性和原权重保留。

前期离线验证的 Bash、Node、根 Compose、H2O Compose 与 Python AST 检查通过；当时尚无完整真实权重，没有启动 H2O/gateway，不能作为 three-probe 或真实推理验收。尝试运行源码 pytest 时，系统 Python 3.13 缺少 vLLM，隔离后端 Python 3.12 缺少 pytest；因此源码 pytest 及项目完整 lint 未通过/未运行。`shellcheck` 不可用。命令输出的重复 FFmpeg Objective-C class 和 Triton 缺失警告来自环境依赖，本次未处理。

此前单向部署/目录改名阶段的夹具与 HTTP 验收不能代替本次真实模型结果；部署验证脚本已删除，不作为部署文件保留。

后续启动排障：固定 revision 模型文件已通过完整校验。共享准备器继承系统代理后，官方基础后端与 H2O 私有 wheel 已完成构建、版本、依赖和原生扩展检查，网关已启动。随后真实 Qwen3.5 加载暴露旧 FP32 head 探针错误调用独立 backbone 的 `NoneType.offset`；补丁改为通过原生 forward 捕获 head 输入，并继续验证 FP32 dtype、adapter 投影一致性和绕过/额外缩放拒绝。真实模型 head 安装及 BF16 权重/lookup 保留通过；6项 tied/untied、bias、softcap、原生初始化与拒绝回滚烟测通过。以上不是统一接口或官方 three-probe 结果。

2026-10-26：根目录强制重建流程的7项隔离回归及5项入口场景通过，部署语法/配置检查通过。两次实际根目录H2O启动均完成依赖模型安全停止、历史目标环境删除及新虚拟环境/pip创建，随后uv获取官方vLLM 0.32.0 wheel连接超时，显式系统代理重试仍失败。半成品为 `~/.local/share/local-model-backend/runtime/3c55218aa0633b9f/`，有安装器标记但无 `.ready`，`current` 不存在；本次未激活后端、未启动H2O/shim，也未执行统一入口推理验收。下次仍使用 `./start.sh h2o`，需先确保官方GitHub Release资源下载链路可用；上文历史结果不代表本次通过。
