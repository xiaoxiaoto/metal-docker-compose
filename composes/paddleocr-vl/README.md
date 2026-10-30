# PaddleOCR-VL · vLLM + vLLM Metal


## 共享优化接入（默认关闭）

本目录使用基础vLLM Metal显式provider合同；扩展功能分包、兼容及验收规则见根 `AGENTS.md`。
启动时固化公共 `METAL_MODEL_OPTIMIZERS`、`METAL_QUANTIZATION`、
`METAL_QUANTIZATION_PROFILE`、`METAL_ANE_PROFILE`、`METAL_STREAMING_PROFILE`；
默认量化 off、provider 列表为空。dynamic 须显式选择 `dynamic_quantization` 并提供绝对路径批准 profile；
ANE 和 streaming 同样须显式选择各自 provider 并提供绝对路径 profile。没有完整真实任务批准时保持 off。


本机 Apple Silicon 使用共享隔离后端的 **vLLM + vLLM Metal**，通过 Metal 原生图像适配执行官方 `PaddlePaddle/PaddleOCR-VL`（0.9B）识别。不使用 Omni、自建 HTTP 服务、CUDA Docker、CPU fallback 或量化转换。

- 官方配方：https://docs.vllm.ai/projects/recipes/en/latest/PaddlePaddle/PaddleOCR-VL.html
- 官方 Metal 支持矩阵：https://github.com/vllm-project/vllm-metal/blob/main/docs/supported_models.md （当前矩阵示例为 1.6；本目录原版 0.9B 仍需真实推理验收，不能把架构适配等同于本机已通过）。
- 规范模型 ID：**`paddleocr-vl`**，不是 `PaddlePaddle/PaddleOCR-VL` 或 `PaddleOCR-VL-0.9B`。下载仓库仍带作者前缀。
- 后端只监听 `127.0.0.1:8098`；公开调用及验收走统一入口 `http://127.0.0.1/v1/`。

根目录将本模型配置为 **dynamic，空闲300秒卸载**。每次根目录启动（包括默认/gateway/lifecycle）先安全停止所有依赖模型、删除历史共享环境和私有 overlay，再重建共享后端；停止失败不删除，构建失败不回退。`./start.sh paddleocr` 每次调用本目录 `start.sh --prepare-only` 重新校验权重并生成 plist，首次推理才离线加载。权重、下载缓存和日志保留；参数以本次设置为准。详见[根目录重建合同](../../README.md#每次根目录启动前重建共享环境)。

冷模型不在聚合列表中，单模型身份查询返回503且不唤醒；先在根目录查看 `python3 -I runtime_mgmt/lifecycle_control.py status`，发送下文真实OCR请求，完成后再检查身份。冷请求另留最多600秒加载预算；不直连8098推理，避免绕过保活计数。

## 环境与准备

Apple Silicon、macOS 15+、Python 3.11+（根控制脚本与启动引导）、ModelScope CLI；网关另需 Docker Desktop。复用根目录 `common_env/prepare_backend.py` 准备的 uv 托管 Python `>=3.12,<3.13` 非 editable 共享环境。共享依赖仅由根目录 [`pyproject.toml`](../../pyproject.toml) / [`uv.lock`](../../uv.lock) 声明，以 `uv sync --locked` 安装：vLLM/Metal `0.32.0`，Omni 固定 `9842ebc17cf1d9c6b440ecb6a79039162f68a1dd`，Omni Metal 来自 `https://github.com/xiaoxiaoto/vllm-omni-metal.git` 的 `5774df7d720b3a8c6d32035729b42c31e480f346`，不依赖相邻源码。保留官方联网预检，不随 main 自动升级，MPS 与 `pip check` 等验证通过后才激活，详见[根目录合同](../../README.md)。本模型只激活 `metal` 插件，不启动 Omni；不额外创建环境或修改 Homebrew。

vLLM 与 vllm-metal 也从官方 GitHub 源码安装，分别固定 `vllm-project/vllm` 的 `5df63a7203cdb9328d824a27b682e77c7b9928be` 与 `vllm-project/vllm-metal` 的 `6d6f2c37f8242f1ff1a336534adf6c2d2fe2936d`（v0.32.0）；由 uv 隔离构建非 editable wheel，不使用官方预编译 wheel。vLLM 的 CPU 编译形态只是 Metal 插件底座，不代表模型 CPU fallback。

`prepare.py` 只选择官方固定仓库 `~/.cache/modelscope/hub/models/PaddlePaddle/PaddleOCR-VL/`，不混选其他版本或作者。完整缓存直接使用；没有或不完整则由 ModelScope 下载/补全 `master`，主权重约 1.92 GB，另有 tokenizer/processor 文件。不下载 `PP-DocLayoutV2/` 和评测产物，不删除已有缓存。校验架构、未量化配置、必要文件、文件路径边界、索引与 safetensors 数据边界，不加载张量、不执行缓存代码或 pickle。

启动参数保留官方关闭 prefix caching、`mm-processor-cache-gb=0` 的策略。Metal 资源预算设为单请求/单图、8192 上下文与批 token、`gpu-memory-utilization=0.15`，保持 bfloat16、spawn、eager 和同步调度。不提高 MPS 内存水位。这是**待真机验收的初始预算**，不是性能或最低内存承诺；大图/长输出可能超过上下文，请先测试小图。

官方配置需要 `--trust-remote-code` 加载仓库的 processor/config；准备阶段只静态检查，启动阶段会执行这些官方仓库代码。请先审阅缓存中的 Python 文件，只使用可信仓库。`master` 不是固定 revision；完整缓存不自动更新，首次下载需自行核对来源。运行时 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，不临时下载权重。

## 启停

```bash
cd composes/paddleocr-vl
./start.sh --prepare-only # 只准备并校验 plist，不加载模型，供根入口自动部署复用
./start.sh  # 原显式启动：自检、准备并加载本机模型；总代理在根目录单独启动
./stop.sh
```

`--prepare-only` 执行完整准备流程，通过 plist 校验后退出，不 bootstrap。已注册任务和端口冲突仍拒绝，非法参数非零退出。根目录每次重新准备本模型部署；只有 resident 路径在启动最后要求聚合列表含指定 ID。health、列表和登记不能代替识别验收。

未提供 `VLLM_BACKEND_PYTHON` 时，本目录启动脚本也调用共享准备器，安全停用全部依赖模型、删除历史环境后重建；只恢复所选目标。`stop.sh` 仍只停止本模型。先确认下载、磁盘与内存预算，勿为单模型验证启动 `all`；不终止占用 8098 的其他进程，加载最多等待600秒，失败只卸载本次启动任务。

launchctl label：`com.local.paddleocr-vl-metal`；plist：`~/Library/Application Support/paddleocr-vl/com.local.paddleocr-vl-metal.plist`，不是登录自启项。日志：`~/Library/Logs/paddleocr-vl-metal.log`、`~/Library/Logs/paddleocr-vl-metal.error.log`。

## 身份检查与真实 OCR

先在根目录启动 `gateway`、`paddleocr`；冷状态先查看生命周期状态并完成首次真实 OCR，再检查聚合列表和单后端身份（加载后两者都应包含 `paddleocr-vl`）：

```bash
curl --fail-with-body --show-error --connect-timeout 5 --max-time 10 \
  'http://127.0.0.1/v1/models'
curl --fail-with-body --show-error --connect-timeout 5 --max-time 10 \
  --get 'http://127.0.0.1/v1/models' --data-urlencode 'model=paddleocr-vl'
```

### 方式一：官方示例图，直接复制 curl

识别这张[官方示例图](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png)，无需本地文件或 base64。`OCR:` 是任务提示；本例需要模型服务能访问可信的图片 URL，离线权重配置不禁止请求图片。

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

终端显示完整 JSON，识别文字位于 `choices[0].message.content`，不保存请求或结果文件。成功需 HTTP 200、模型身份匹配、非空文字和 `finish_reason=stop`，还须与图片对照；`length` 表示输出截断。单图 OCR 不等于逐字准确率或完整 PDF 版面解析。

### 方式二：本地 PNG

将下面 `/实际路径/清晰文档.png` 换成**存在的真实 PNG**，先用清晰、短文本小图。图片用 data URL 嵌入 JSON（非 multipart），避免让后端访问任意远端地址；整个 JSON（含 base64）不得超过 25 MiB。

```bash
set -o pipefail
curl --fail-with-body --show-error --connect-timeout 5 --max-time 10 \
  'http://127.0.0.1/v1/models'

curl --fail-with-body --show-error --connect-timeout 5 --max-time 10 \
  --get 'http://127.0.0.1/v1/models' --data-urlencode 'model=paddleocr-vl'

python3 -c 'import base64,json,sys; from pathlib import Path; image=Path(sys.argv[1]).read_bytes(); print(json.dumps({"model":"paddleocr-vl","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,"+base64.b64encode(image).decode()}},{"type":"text","text":"OCR:"}]}],"temperature":0,"max_tokens":1024,"stream":False}, ensure_ascii=False))' '/实际路径/清晰文档.png' | \
  curl --fail-with-body --show-error --connect-timeout 5 --max-time 1800 \
    'http://127.0.0.1/v1/chat/completions' \
    --header 'Content-Type: application/json' --data-binary @- | \
  python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin), ensure_ascii=False, indent=2))'
```

JSON直接在终端展示，不生成含图片或识别内容的临时文件。输入必须是存在的实际PNG；
整个JSON含base64不得超过25MiB，先用短文本小图。检查HTTP状态、模型身份、
非空识别文本和正常结束，并与图片文字比对；缩进展示不等于识别准确。

其他官方任务提示分别为 `Table Recognition:`、`Formula Recognition:`、`Chart Recognition:`，放在同一 `text` 字段。这是单图/元素识别 API，**不是完整 PDF 排版还原流水线**。PP-DocLayoutV2/PaddleOCR 文档解析需要额外独立环境及客户端模型名配置，本目录不部署；不使用配方里的旧别名绕过本项目命名规范。

