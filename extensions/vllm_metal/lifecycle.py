# SPDX-License-Identifier: Apache-2.0
"""Deterministic ownership for load-time optimizer resources."""


def register_resource(model, resource):
    """Keep a closeable resource outside model parameters and audit metadata."""
    if not callable(getattr(resource, "close", None)):
        raise TypeError("an optimizer resource must provide close()")
    resources = getattr(model, "_metal_extension_resources", None)
    if resources is None:
        resources = []
        object.__setattr__(model, "_metal_extension_resources", resources)
    if type(resources) is not list:
        raise TypeError("optimizer resource ownership was replaced")
    if any(item is resource for item in resources):
        raise ValueError("optimizer resource already registered")
    resources.append(resource)
    return resource


def close_model_optimizers(model):
    """Close every registered resource in reverse order, including after errors."""
    if model is None:
        return
    resources = getattr(model, "_metal_extension_resources", None)
    if resources is None:
        return
    if type(resources) is not list:
        raise TypeError("optimizer resource ownership was replaced")
    pending = resources[:]
    resources.clear()
    failures = []
    for resource in reversed(pending):
        try:
            resource.close()
        except Exception as error:
            failures.append(error)
    if failures:
        raise ExceptionGroup("closing model optimizers failed", failures)
