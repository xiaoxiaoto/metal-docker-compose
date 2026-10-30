# WeVisOne-VL-4B · vLLM + vLLM Metal

本目录在 Apple Silicon Mac 上以 **vLLM + vLLM Metal** 运行 ModelScope 固定发布的 `tencent-community/WeVisOne-VL-4B`。7 个任务统一由 `POST /v1/chat/completions` 的 `wevisone_task` 分流；模型私有 endpoint / worker 插件复用同一已加载主模型，提供检测、定位、计数、OCR、分割与抠图。不使用 Omni、独立 HTTP shim、CUDA Docker、第二份 Transformers 主模型或 CPU 推理回退。

- 规范模型 ID：**`wevisone-vl-4b`**；根目录目标：`wevisone`。
- 本机后端：`127.0.0.1:8101`；公开调用与验收统一使用 `http://127.0.0.1/v1/`。
- 原生架构：`Qwen3VLForConditionalGeneration` / `qwen3_vl`，BF16，词表大小 `152680`；无需 `--trust-remote-code`。
- 统一端点：`POST /v1/chat/completions` + `wevisone_task`；像素任务由 MLX BF16 主模型特征 → Torch-MPS FP32 官方 `mask_decoder.pt` 解码，返回逐对象 mask、alpha、RGBA 抠图与叠加图；旧 `POST /v1/wevisone/predict` 已弃用。

## 官方来源与固定缓存

- ModelScope 固定发布：[`tencent-community/WeVisOne-VL-4B`](https://modelscope.cn/models/tencent-community/WeVisOne-VL-4B)，revision：`855d2a73ac0b929f668e6ac4f5deccc667aba99b`；上游[官方模型卡](https://huggingface.co/tencent/WeVisOne-VL-4B)。
- [官方任务提示源码](https://github.com/Tencent/WeVisOne-VL/blob/9a6bcd85404505df0d05cfb5d97512ccacf35dee/wevisone_vl/prompts.py)与[官方推理语义](https://github.com/Tencent/WeVisOne-VL/blob/9a6bcd85404505df0d05cfb5d97512ccacf35dee/wevisone_vl/engine.py)，源码 commit：`9a6bcd85404505df0d05cfb5d97512ccacf35dee`。私有 wheel 只纳入经固定摘要校验的四个官方像素模块及许可证，不安装官方 engine，不执行模型缓存代码。

该 ModelScope 仓库的八个部署文件与上游 `tencent/WeVisOne-VL-4B` 发布逐字节一致（SHA256 相同），缓存使用固定布局：

```text
~/.cache/modelscope/hub/models/tencent-community/WeVisOne-VL-4B/
```

准备器只从上述固定 revision 下载白名单文件，不扫描其他作者、版本、量化版或任意模型目录，也不跟随 `master` 自动升级。目录中已有的 `LICENSE`、`README.md`、`configuration.json` 及 `.msc` / `.mdl` 元数据不执行、不删除。

模型文件白名单：`config.json`、`generation_config.json`、`processor_config.json`、`tokenizer_config.json`、`tokenizer.json`、`chat_template.jinja`、`model.safetensors`、`mask_decoder.pt`。缺失或摘要不符时，仅从固定 revision 补全对应文件，先写同目录临时文件，校验通过后才替换；完整模型缓存不联网下载。所有文件核对固定大小与 SHA256，主权重还检查 safetensors 头、BF16 张量形状、连续偏移、数据长度与完整参数量，不加载张量。拒绝缓存目录或文件符号链接、大小/摘要不符及混入其他 safetensors 主权重。像素 sidecar 只有在固定摘要校验通过后才由 `torch.load(weights_only=True)` 读取；不执行任意 pickle。

官方主权重大小为 **9,661,250,408 bytes（约9.66 GB / 9.00 GiB）**，共 `4,830,581,248` 个 BF16 参数，SHA256：`002c3ac0783af7c86dc28b282c84dd4bcd0c29dcb3b5e0d584492fe3212acb4a`。tokenizer 另约11.6 MB；`mask_decoder.pt` 为 **38,534,741 bytes**，SHA256：`8528460481bd09d0780c53559b08194c7abbd29dff0d51783b619158acbe7b46`。每次准备流式读取权重核对摘要，需预留读盘时间；磁盘大小不等于含特征重算与像素解码的运行峰值。主模型强制 `--load-format safetensors`，sidecar 严格加载官方 decoder state。

官方 `config.json` 使用 transformers 5.x 的 `text_config.rope_parameters` 与 `vision_config.model_type=qwen3_vl_vision`，mlx_vlm 0.7.6 使用旧式 `rope_theta` / `rope_scaling` 和 `qwen3_vl`。准备器生成只读运行时视图 `~/Library/Application Support/wevisone-vl-4b/model/`：`config.json` 从官方配置派生，只改字段名/类型标签，其余七个文件符号链接到已校验缓存；官方缓存与摘要不变。plist 指向该视图，每次准备原子重建。

## 环境、资源与公共优化

需要 Apple Silicon、macOS 15+、Python 3.11+（启动引导）、launchctl；网关另需 Docker Desktop。共享后端按根 `pyproject.toml` / `uv.lock` 准备，见[根目录合同](../../README.md)。`prepare.py` 使用标准库 HTTP 客户端下载固定 ModelScope 文件；`prepare_runtime.py` 在已校验共享环境之上创建模型私有隔离 overlay，下载固定官方像素源码、构建并安装非 editable `wevisone-vl-metal` wheel，不修改共享 site-packages。

基础 vLLM 的 macOS CPU 构建形态只是 Metal 插件底座，不是模型 CPU fallback。启动检查 MLX Metal 与 Torch MPS，设置 `VLLM_PLUGINS=metal,wevisone_vl_metal`、`PYTORCH_ENABLE_MPS_FALLBACK=0`；不可用就失败，不切 CPU。

启动使用 BF16、`spawn`、eager、同步调度、关闭 prefix caching、`mm-processor-cache-gb=0`；单图、单个生成序列，禁用视频/音频，8192上下文与批 token，`gpu-memory-utilization=0.35`。图像处理预算为 `min_pixels=16384`、`max_pixels=2073600`，沿用官方 engine，而非 processor 的超大默认值。图像 token、提示与生成共享上下文；像素解码还需要重算特征及1024×1024图像分支，不承诺最低内存或吞吐。原图不超过4,194,304像素，先用小图、短输出。

公共优化参数仍采用统一 `METAL_*` 合同，但当前像素插件只接受默认空 provider、量化 `off`，明确拒绝 dynamic quantization、ANE、streaming、LoRA、speculative 和多 worker；没有兼容别名、静默覆盖或 CPU fallback。生命周期的 dynamic 仅指按需加载，不是量化。

私有源码集中在 `extensions/vllm/wevisone_vl/`，wheel 映射为 `wevisone_vl_metal`，通过 `vllm.endpoint_plugins` 及 `--worker-extension-cls` 接入。overlay 位于 `~/.local/share/local-model-backend/overlays/wevisone-vl-4b/<指纹>/`，校验固定共享来源、依赖版本、wheel / 安装文件 / RECORD、entry point 与 `pip check` 后才发布；当前兼容组为 vLLM `0.32.0+cpu`、vLLM Metal `0.32.0`、mlx-vlm `0.7.6`。源码或共享环境身份变化须重建并重新验收，不能修改已安装 site-packages 或放宽版本门禁。

特征重算由同一 Metal worker 的 collective RPC 同步执行，不进入 paged-attention 上下文，结束恢复临时 vision merger 捕获器；MLX / MPS 通过共享 bridge 传递数组。像素解码器保持官方结构与 FP32，只对该实例的非整除 adaptive pooling 使用同一 floor/ceil 分箱的 MPS 实现，不全局 monkeypatch，不回退 CPU。像素请求取消后等待 worker RPC 清理再释放串行锁，模型随原 launchctl 生命周期整体卸载。

## 启停、动态加载与离线边界

在根目录使用：

```bash
./start.sh wevisone
python3 -I runtime_mgmt/lifecycle_control.py status
./stop.sh wevisone
```

根入口将本模型按 **dynamic、空闲300秒卸载** 管理，启动预算600秒，加载完成后的请求预算1800秒，总代理超时2400秒。目标准备好不等于模型已驻留；首次推理通过统一入口唤醒，冷状态不出现在聚合模型列表中，单模型身份查询返回503且不会唤醒。完成首次真实请求后再核对身份。不要直连8101做业务推理，否则绕过生命周期保活计数；直连仅用于后端故障定位。

**每次根目录启动，包括 gateway/lifecycle，都会安全停止依赖模型并重建共享后端，可能中断其他模型在途请求。** 历史共享环境及私有 overlay 按根合同清理；停止失败不删除、构建失败不回退，权重、下载缓存和日志保留。只恢复所选目标，勿为单模型验收启动 `all`。详见[根目录重建合同](../../README.md#每次根目录启动前重建共享环境)。

独立入口与 PaddleOCR 部署模式相同：

```bash
cd composes/wevisone-vl-4b
./start.sh --prepare-only # 准备后端、私有 wheel、模型视图和 plist；不 bootstrap 模型
./start.sh                # 显式直接加载，总代理仍由根目录管理
./stop.sh
```

未提供 `VLLM_BACKEND_PYTHON` 时，独立启动也会调用共享准备器，产生上述停止与重建影响。`--prepare-only` 不是“不联网”：缺失权重仍会下载；运行时设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，只离线加载已校验文件。URL 图片仍可能需要网络；完全离线请求应发送本地图片 data URL。已有已注册任务或端口冲突会明确拒绝，不自动杀进程。加载最多等待600秒，失败只卸载本次启动的任务。

已选定且通过来源校验的共享环境可显式复用，避免为更新模型私有插件而重建共享后端：

```bash
# 从项目根目录执行；先停止 WeVisOne，在途请求会中断，其他模型不停止。
./stop.sh wevisone
VLLM_BACKEND_PYTHON="$HOME/.local/share/local-model-backend/current/bin/python" \
  ./composes/wevisone-vl-4b/start.sh --prepare-only
python3 -I runtime_mgmt/lifecycle_control.py check-dynamic wevisone
```

此入口只准备 WeVisOne；动态模型下一次统一入口推理会加载新 plist。不要用根 `./start.sh gateway` 代替私有更新。

- label：`com.local.wevisone-vl-4b-metal`
- plist：`~/Library/Application Support/wevisone-vl-4b/com.local.wevisone-vl-4b-metal.plist`，不是登录自启项。
- 日志：`~/Library/Logs/wevisone-vl-4b-metal.log`、`~/Library/Logs/wevisone-vl-4b-metal.error.log`。

## 统一 chat 接口

七个任务统一使用 `POST /v1/chat/completions`，由请求体顶层 `wevisone_task` 分流；不带该字段时请求原样透传，行为与此前一致（包括流式 SSE），响应不附加 `pixel`。

| `wevisone_task` | `predictions` | `artifacts`（`objects>0` 时） |
|---|---|---|
| `detection` | 逐框 | `overlay.png` |
| `referring` | 指代表达单框 | `overlay.png` |
| `pointing` | 逐点 | `overlay.png` |
| `counting` | 逐框计数 | `overlay.png` |
| `ocr` | 逐文本框，`label` 为文字 | `overlay.png` |
| `segmentation` | 逐框 | `overlay.png`、`masks/mask_NNN.png`、`alpha.png`、`cutout.png`、`segmentation.png` |
| `matting` | 逐框 | `overlay.png`、`masks/mask_NNN.png`、`alpha.png`、`cutout.png` |

请求只接受一组 `messages`：可选 system（缺省 `You are a helpful assistant.`）、恰一个 user；content 中恰一个 `image_url`（HTTPS 或 JPEG/PNG data URL，不接受 HTTP、重定向、本地路径、多图或视频）和一段非空文本（≤4096字符）。该文本是查询词，服务端按固定源码 `9a6bcd8...` 的官方模板生成提示；`ocr` 使用固定提示，文本仍须非空但不参与推理。`max_tokens` / `max_completion_tokens` 为1…4096（默认2048），`stream` 必须为 false；其余采样参数透传，缺省为 `temperature=0`、`top_p=0.95`、`top_k=50`、`repetition_penalty=1.05`，`stop` 与 `skip_special_tokens` 由服务端固定。

响应是标准 `chat.completion`：`message.content` 保留原始对象标记，`message.pixel` 恒存在，包含 `task`、`image_size`、`objects`、`predictions`（原图像素坐标，保留1位小数）和 `artifacts`。图像为不带 `data:` 前缀的 base64 PNG，键名固定白名单；产物无状态内嵌，不返回下载链接，也不在服务端落盘或缓存，已返回的数据不随模型卸载或进程重启失效；`objects=0` 时 `predictions` 与 `artifacts` 均为空。坐标越界、框无面积、`<OBJ>` 与掩码数量不一致或生成截断都返回错误，不产生伪产物；最多16个对象。错误为纯文本：400（参数、图片获取、截断）、404（模型不符）、413（请求/图片 >25 MiB 或 >4,194,304像素）、415（非 JSON）、499（客户端断开）、500（特征或解码失败）。

实现边界：

- `WeVisOneChatMiddleware` 只在 `POST /v1/chat/completions` 上缓冲请求体（上限25 MiB），先做字节级 `"wevisone_task"` 预筛；未命中原样回灌 `receive`，普通 chat 与 SSE 不改变行为。中间件在 endpoint 插件阶段注册，位于 vLLM 鉴权/CORS 之后，是最内层中间件；top-level 无 `wevisone_task` 的误命中也会回灌。
- 提示词、标记解析、坐标换算在 `extensions/vllm/wevisone_vl/tasks.py`，对象级叠加图在 `render.py`：与旧客户端相同的颜色顺序（red/lime/dodgerblue/orange/magenta/cyan）、`max(2, round(W/400))` 线宽、`max(6, round(W/85))` 点半径、14px 标签偏移和 PIL 默认位图字体，同一 Pillow 版本下 PNG 逐字节一致。
- 像素任务仍由同一 worker 的 collective RPC 解码，串行锁只包解码阶段；客户端断开时先等待 worker RPC 清理再释放锁。划分/抠图的 `overlay.png` 由 worker 产出，避免两套像素语义。
- 旧 `POST /v1/wevisone/predict` 保留一个 minor 版本，响应头带 `Deprecation: true` 与 `Link: </v1/chat/completions>; rel="successor-version"`，格式不变。

## 内嵌 save_result 后处理

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

## 统一入口身份检查与官方检测请求

先运行根目录 `./start.sh wevisone`。下列例子使用固定官方源码中的 [boys.jpg](https://github.com/Tencent/WeVisOne-VL/blob/9a6bcd85404505df0d05cfb5d97512ccacf35dee/demo/examples/images/boys.jpg)，检测 `person`。冷模型先执行聚合查询确认网关，再由检测请求首次唤醒；随后单模型身份检查必须包含 `wevisone-vl-4b`，必要时再查一次聚合列表。

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

核对每个框覆盖画面中的人物及数量，并打开 `/tmp/wevisone-detection/overlay.png` 对照原图。

### 指代表达定位

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

检查返回框覆盖最左侧打鼓的男孩，而非整张图片。

### 指点

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

检查点落在沙发靠枕上；点坐标不是边界框。

### 计数

`counting` 使用官方框计数提示，`objects` 即总数。图片有四个靠枕；核对四个有效框及数量，不把沙发坐垫计为靠枕。

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

### OCR

`ocr` 使用官方固定提示 `detect all the word in the box format`，请求文本仍须非空但不参与推理；`label` 是识别出的文字。逐字对照招牌的 REAL COFFEE、Tea Café、10 METERS、AHEAD、TURN RIGHT、IN、SMALL ALLEY 等文字与定位；艺术字和重音符号可能识别有误。

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

### 分割与抠图

`segmentation` 与 `matting` 使用官方 alpha-matte 提示，返回真实像素结果：服务端先由原生 chat 生成 `<OBJ>`，再在同一 Metal worker 中用已加载的 MLX BF16 主模型重算无 KV 缓存的特征，提取最终 RMSNorm 前的对象状态和四层 ViT merger 输入，官方像素解码器以 Torch-MPS FP32 执行。没有第二份 Transformers 主模型或 CPU 回退，框坐标不能代替真实掩码。

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

输出在 `/tmp/wevisone-segmentation/`：`response.txt`、`result.json`、`masks/mask_000.png` 等逐对象连续掩码、`alpha.png`、`cutout.png`（RGBA）、`overlay.png` 和 `segmentation.png`。最后一项以浮点概率 `>=0.5` 阈值化，只含0/255；掩码及 alpha 保留8位连续灰度。核对尺寸与原图一致、有效前景／背景和人物实际轮廓。

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

抠图与分割使用同一提示与解码器，但保留连续 alpha、不生成二值 `segmentation.png`。核对 `cutout.png` 的透明通道等于 alpha、RGB 与原图一致，并检查发丝、半透明边缘和背景泄漏；8位 PNG 会量化概率。

## 普通 chat 直通（无 `wevisone_task`）

不设置 `wevisone_task` 的请求原样交给模型，响应没有 `pixel`，不能套用上面的 `save_result`；流式 SSE 行为不变。文字定位使用官方 `ocr_grounding` 提示：

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

点计数或其它自由提示同样不带 `wevisone_task`，只返回文本；需要结构化框/点与叠加图时改用对应任务。

## 验收边界

部署验收应经统一入口分别核对 7 个 `wevisone_task`、负例状态码、普通 chat（含 SSE）与旧路由弃用头；图像任务必须对照原图与逐对象输出，不能用准备成功、health、模型列表或框坐标代替像素质量。检查冷启动、空闲卸载、再次加载、像素并发/断连后的锁释放，以及停止后的进程与端口释放。

模型接口兼容、数值对照或单张示例不代表完整质量基准；没有标注数据集时不声明分割 IoU、alpha MAD、多人重叠、极细发丝或半透明材质精度。模型生成可能漏检、拒识或截断，`objects=0` 是无目标结果，不是分割成功；像素返回为8位 PNG，不是浮点张量接口。不支持带 `wevisone_task` 的流式输出、批量、多 worker 或任意优化组合；CJK 标签仍使用 PIL 默认位图字体，可能无法渲染。
