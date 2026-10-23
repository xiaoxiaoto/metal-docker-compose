#!/usr/bin/env python3
"""真实 MPS 采样/生命周期与随机 tiny NAR/VAE 自检；不加载或下载大权重。"""

import os
import sys
from importlib.metadata import version
from types import SimpleNamespace


def sampler_model(torch, y):
    model = object.__new__(y.Yue2ForCausalLM)
    torch.nn.Module.__init__(model)
    model._states = {}
    model._metal_ffn = None
    model._pending = model._stage = model._pinned = model._step_discard = None
    model._step_rows = []
    model._decode_t0 = {}
    model._audio_queue = []
    model._last_mm = None
    model._max_model_len = 128
    model._synthesis = y._SynthesisQueue()
    return model


def prepare_step(torch, y, model, seeds, step, max_frames=16):
    ids = torch.ones(len(seeds), dtype=torch.int32, device="mps")
    model.prepare_runner_inputs(
        req_ids=[str(seed) for seed in seeds],
        num_computed_tokens=[step] * len(seeds),
        num_scheduled_tokens=[1] * len(seeds),
        input_ids=ids,
        positions=ids - 1,
        sampling_extra_args=[
            {
                y.KEY_SEED: seed,
                y.KEY_PREFIX_IDS: [1],
                y.KEY_MIN_TOKENS: 0,
                y.KEY_MAX_AUDIO_FRAMES: max_frames,
                y.KEY_SKIP_SYNTHESIS: True,
            }
            for seed in seeds
        ],
    )


def check_sampler(torch, y):
    logits = torch.randn((1, y.VOCAB_SIZE), generator=torch.Generator().manual_seed(7)).to("mps")
    logits[:, y.MUSIC_END] = -torch.inf

    def draws(seeds):
        model = sampler_model(torch, y)
        result = []
        for step in range(4):
            prepare_step(torch, y, model, seeds, step)
            output = model.sample(logits.expand(len(seeds), -1), None)
            assert output.sampled_token_ids.device.type == "mps"
            result.append(output.sampled_token_ids.cpu())
        assert model._pending[0] is None, "MPS 读回仍使用 CUDA event"
        assert not model._stage.is_pinned() and not model._pinned.is_pinned()
        model._resolve_pending()
        assert all(len(state.history) == 4 for state in model._states.values())
        assert all(state.generator.device.type == "mps" for state in model._states.values())
        model.on_requests_finished(list(model._states))
        assert not model._states and model._pending is None
        return torch.cat(result, dim=1)

    alone = draws([42])
    assert torch.equal(alone, draws([42])), "同 seed 采样不一致"
    assert torch.equal(alone, draws([9, 42])[1:]), "合批改变请求的 RNG/采样"

    model = sampler_model(torch, y)
    fixed = torch.full_like(logits, -torch.inf)
    fixed[:, y.CODEC_OFFSET + 7] = 0
    for step in range(3):
        prepare_step(torch, y, model, [42], step, max_frames=2)
        output = model.sample(fixed, None)
    state = model._states["42"]
    assert output.sampled_token_ids.item() == y.MUSIC_END
    assert state.finished and state.truncated and len(state.history) == 2
    model.on_requests_finished(["42"])

    prepare_step(torch, y, model, [42], 0)
    model.sample(torch.full_like(logits, -torch.inf), None)
    try:
        model._resolve_pending()
    except RuntimeError as error:
        assert "all -inf" in str(error)
    else:
        raise AssertionError("非法采样分数未拒绝")


def check_synthesis(torch, y):
    # Random tiny modules exercise the real upstream NAR kernels, not a synthetic audio server.
    torch.manual_seed(42)
    nn = torch.nn
    model = sampler_model(torch, y)
    config = SimpleNamespace(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rms_norm_eps=1e-6,
    )
    model.config = config
    model.vocab_size = y.VOCAB_SIZE
    model.max_position_embeddings = model.max_latent_frames = 128
    model._timestep_shift = 1.0
    model.vae2llm = nn.Linear(64, 32)
    model.llm2vae = nn.Linear(32, 64)
    model.time_embedder = y._TimestepEmbedder(32)
    model.latent_pos_embed = y._AudioPositionEmbedding(128, 32)
    model.rotary_emb = y._RotaryEmbedding(8)
    model.nar_layers = nn.ModuleList([y._NARLayer(config)])
    model.nar_fused = None
    model.model = nn.Module()
    model.model.embed_tokens = nn.Embedding(y.VOCAB_SIZE, 32)
    model.model.norm = y._RMSNorm(32)
    layer = nn.Module()
    layer.input_layernorm = y._RMSNorm(32)
    layer.self_attn = nn.Module()
    layer.self_attn.num_heads = 4
    layer.self_attn.num_kv_heads = 2
    layer.self_attn.head_dim = 8
    layer.self_attn.qkv_proj = nn.Linear(32, 64, bias=False)
    layer.self_attn.q_norm = y._RMSNorm(8)
    layer.self_attn.k_norm = y._RMSNorm(8)
    model.model.layers = nn.ModuleList([layer])
    model.to(device="mps", dtype=torch.bfloat16).eval()
    prefix, codec = [y.EOD, y.ABC_START, y.ABC_END, y.MUSIC_START], [1, 2, 3, 4]
    with torch.inference_mode():
        latents = y.synthesize(model, prefix, codec, 42, steps=2)
        assert torch.isfinite(latents).all() and latents.shape == (4, 64)
        assert torch.equal(latents, y.synthesize(model, prefix, codec, 42, steps=2))
        from vllm_omni_metal.ane.runtime import DenseSwiGLU

        model._metal_ffn = DenseSwiGLU(
            [(w[6], w[7]) for w in model.nar_weights()],
            scope="tiny-nar-self-check",
            precision="fp16",
        )
        assert torch.equal(latents, y.synthesize(model, prefix, codec, 42, steps=2))
        assert model._metal_ffn.metrics["native_calls"] > 0
        model.close_metal_operators()
        assert model._metal_ffn is None
        vae_config = y.YuE2VAEConfig(
            decoder_config=dict(
                out_channels=2,
                channels=2,
                c_mults=[1, 2],
                strides=[8, 8],
                latent_dim=64,
                use_snake=True,
                use_filter=False,
            ),
            downsampling_ratio=64,
        )
        model.__dict__["_vae"] = y.YuE2VAE(vae_config, decoder_only=True).float()
        model._vae_device = None
        audio = model._decode_latents(latents)
        assert model._vae.decoder_device.type == "mps", "VAE 仍退回 CPU"
        assert next(model._vae.parameters()).dtype == torch.float32
        assert audio.shape[0] == 2 and audio.shape[1] > 0
        assert audio.device.type == "cpu" and torch.isfinite(audio).all() and audio.abs().max() > 0
        try:
            model._vae.decode(torch.full((1, 64, 4), float("nan")))
        except ValueError:
            pass
        else:
            raise AssertionError("VAE 未拒绝非有限 latents")
    torch.mps.synchronize()


def check():
    if os.environ.get("YUE2_METAL_QUANTIZATION", "off") != "off":
        raise RuntimeError("随机 tiny 自检必须显式关闭量化；不是 dynamic profile 验收")
    if "yue2mps2" not in version("vllm-omni"):
        raise RuntimeError("需要已应用 yue2-mps.patch 的 yue2mps2 私有 Omni overlay")
    import torch
    from vllm_omni.model_executor.models.yue2 import yue2 as y

    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("请取消 PYTORCH_ENABLE_MPS_FALLBACK，不能以 CPU fallback 通过 MPS 自检")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS 不可用")
    check_sampler(torch, y)
    check_synthesis(torch, y)


if __name__ == "__main__":
    try:
        check()
    except Exception as error:
        print(f"拒绝启动 YuE2 Metal：{error}\n未下载权重、未注册模型任务。", file=sys.stderr)
        sys.exit(1)
    print("YuE2 MPS 自检通过：seed/合批隔离、同步读回、终止与清理、tiny NAR、FP32 MPS VAE。")
    print("随机小权重自检不是实际 YuE2-3B 歌曲验收。")
