# SPDX-License-Identifier: Apache-2.0
"""Qwen gate-only ANE controller; down remains owned by DynamicEngine."""

import hashlib
import json

import torch

from vllm_omni_metal.ane.cache import profile_label
from vllm_omni_metal.ane.projection import (
    DenseProjection,
    prepare_projection,
    publish_projection_profile,
)


class GateProjection(torch.nn.Module):
    def __init__(self, source, *, scope, prepare):
        super().__init__()
        if source._forward_hooks or source._forward_pre_hooks or source._backward_hooks:
            raise ValueError("ANE gate cannot discard projection hooks")
        if type(source) is not torch.nn.Linear:
            from vllm.model_executor.layers.linear import (
                ColumnParallelLinear,
                UnquantizedLinearMethod,
            )

            if (
                type(source) is not ColumnParallelLinear
                or source.tp_size != 1
                or source.tp_rank != 0
                or type(source.quant_method) is not UnquantizedLinearMethod
            ):
                raise ValueError("ANE gate requires an unquantized complete TP1 projection")
        self.scope = scope
        self.source = source
        self.return_bias = getattr(source, "return_bias", False)
        self.skip_bias_add = getattr(source, "skip_bias_add", False)
        bias = None if self.skip_bias_add else source.bias
        if prepare:
            prepare_projection(source.weight, bias, scope=scope)
        self.projection = DenseProjection(
            source.weight, bias, scope=scope, precision="fp16", _benchmark=prepare
        )
        if self.projection.disabled or self.projection._closed or self.projection._bridge is None:
            self.projection.close()
            raise RuntimeError("ANE gate requires a healthy prepared NE projection")
        self.context = None
        self.shapes = set()
        self.chunks = 0 if prepare else None
        self.hits = 0

    def forward(self, hidden):
        def native(value):
            result = self.source(value)
            return result[0] if self.return_bias else result

        rows = hidden.numel() // hidden.shape[-1]
        self.shapes.add((rows, self.context))
        before = self.projection.metrics.get("ane_chunks", 0)
        result = self.projection(hidden, native=native, context=self.context)
        hits = self.projection.metrics.get("ane_chunks", 0) - before
        self.hits += hits
        if self.chunks != 0 and not hits:
            raise RuntimeError("Qwen ANE gate executed no NE chunks; refusing silent GPU fallback")
        bias = self.source.bias if self.skip_bias_add else None
        return (result, bias) if self.return_bias else result


class ProjectionGroup:
    def __init__(self, blocks, layers, *, scope, prepare):
        self.gates = []
        self._benchmark = prepare
        self._audit = prepare
        try:
            for index in layers:
                ffn = blocks[index].img_mlp
                if isinstance(ffn.gate_layer, GateProjection):
                    raise ValueError("ANE gate already bound")
                self.gates.append(
                    GateProjection(ffn.gate_layer, scope=f"{scope}:gate:{index}", prepare=prepare)
                )
        except BaseException:
            self.close()
            raise
        for index, gate in zip(layers, self.gates, strict=True):
            blocks[index].img_mlp.gate_layer = gate

    @property
    def metrics(self):
        result = {}
        for gate in self.gates:
            for key, value in gate.projection.metrics.items():
                result[key] = result.get(key, 0) + value
        return result

    @property
    def checks(self):
        return {
            (i, *key): value
            for i, gate in enumerate(self.gates)
            for key, value in gate.projection.checks.items()
        }

    @property
    def disabled(self):
        return any(g.projection.disabled or g.projection._closed for g in self.gates)

    def context(self, request):
        context = int(
            hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:7], 16
        )
        for gate in self.gates:
            gate.context = context

    def reset(self, chunks):
        if not self._benchmark:
            raise ValueError("production ANE gate cannot change benchmark chunks")
        for gate in self.gates:
            gate.projection.set_benchmark_chunks(chunks)
            gate.projection.metrics.clear()
            gate.shapes.clear()
            gate.chunks = chunks
            gate.hits = 0

    def finish_audit(self):
        # DenseProjection retains its mandatory live-input full-output audits.
        if any(not gate.projection.checks or gate.hits <= 0 for gate in self.gates):
            raise RuntimeError("every gate must pass live-output audit with actual ANE hits")
        for gate in self.gates:
            gate.projection.finish_audit()
        self._audit = False

    def approve(self, baseline, candidates):
        records = []
        for gate in self.gates:
            gate.staged_policy = {
                "schema": 1,
                "group": gate.projection.identity,
                "device": gate.projection._bridge.device,
                "kind": "linear-bias-v1",
                "profiles": {},
            }
            for rows, context in sorted(gate.shapes):
                record = publish_projection_profile(
                    gate.projection, rows, baseline, candidates, context=context, publish=False
                )
                if record["chunks"] <= 0:
                    raise ValueError("ANE gate did not earn fixed 3% full-pipeline benefit")
                gate.staged_policy["profiles"][profile_label(rows, context)] = record
                records.append(
                    {
                        **record,
                        "gate_scope": gate.scope,
                        "identity": gate.projection.identity,
                        "rows": rows,
                        "context": context,
                        "staged_policy": gate.staged_policy,
                    }
                )
        if not records:
            raise ValueError("no live ANE gate shapes to approve")
        return records

    def production(self):
        replacements = []
        try:
            for gate in self.gates:
                replacements.append(
                    DenseProjection(
                        gate.source.weight,
                        None if gate.skip_bias_add else gate.source.bias,
                        scope=gate.scope,
                        precision="fp16",
                        _staged_policy=gate.staged_policy,
                    )
                )
                if (
                    replacements[-1].disabled
                    or replacements[-1]._closed
                    or replacements[-1]._bridge is None
                ):
                    raise RuntimeError("approved ANE gate policy failed production reload")
            for gate, replacement in zip(self.gates, replacements, strict=True):
                gate.projection.close()
                gate.projection = replacement
                gate.chunks = None
            self._benchmark = False
        except BaseException:
            for replacement in replacements:
                replacement.close()
            raise

    def validate_records(self, records):
        expected = {gate.scope: gate for gate in self.gates}
        covered = set()
        for record in records:
            gate = expected.get(record.get("gate_scope"))
            if (
                gate is None
                or record.get("identity") != gate.projection.identity
                or gate.projection.profiles.get((record.get("rows"), record.get("context")))
                != record.get("chunks")
                or record.get("chunks", 0) <= 0
            ):
                raise ValueError(
                    "joint ANE policy does not match approved gate/shape/context/chunks"
                )
            covered.add(gate.scope)
        if covered != set(expected):
            raise ValueError("joint approval must cover every bound gate")

    def close(self):
        for gate in self.gates:
            gate.projection.close()
