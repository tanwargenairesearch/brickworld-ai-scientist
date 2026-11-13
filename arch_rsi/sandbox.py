"""Static and runtime checks for candidate architecture files, and loading them.

Static: AST allowlist of imports, banned names and attributes, top-level statements limited to
imports, defs, classes and constant assignments. Runtime (in the calling process, which is always a
separate trial or check process): build, parameter cap, shapes, finiteness, forward+backward time.
"""
import ast
import hashlib
import math
import time
import types

import torch

ALLOWED_IMPORTS = {"torch", "torch.nn", "torch.nn.functional", "math"}
BANNED_NAMES = {"eval", "exec", "compile", "open", "getattr", "setattr", "delattr", "globals", "locals", "vars",
                "__import__", "input", "breakpoint", "exit", "quit", "help", "memoryview", "__builtins__",
                "__loader__", "__spec__", "__file__"}
BANNED_TORCH = {"load", "save", "hub", "jit", "compile", "utils", "ops", "classes", "library", "backends", "cuda",
                "mps", "distributed", "multiprocessing", "autograd", "random", "manual_seed", "seed", "initial_seed",
                "set_num_threads", "set_num_interop_threads", "set_default_dtype", "set_default_device",
                "set_default_tensor_type", "set_grad_enabled", "set_printoptions", "set_rng_state", "get_rng_state",
                "use_deterministic_algorithms", "set_deterministic_debug_mode", "set_flush_denormal",
                "from_file", "storage", "package", "profiler", "onnx", "export", "fx", "func", "overrides",
                "testing", "futures", "monitor", "sparse", "quantization", "ao", "nested", "masked", "xpu",
                "set_float32_matmul_precision", "Generator", "fork_rng", "inference_mode",
                "Storage", "UntypedStorage", "TypedStorage", "serialization", "_C"}
BANNED_ATTRS = {"register_forward_pre_hook", "register_full_backward_pre_hook", "register_load_state_dict_post_hook",
                "register_state_dict_pre_hook", "load_state_dict", "state_dict", "share_memory", "numpy", "tolist",
                "item", "data_ptr", "untyped_storage", "storage", "cuda", "to_sparse", "apply_", "map_",
                "format", "format_map"}
SAFE_DUNDERS = {"__init__"}


class CandidateError(ValueError):
    pass


def _torch_chain(node):
    """For an attribute chain rooted at a name, return [root, attr1, attr2, ...]."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return parts[::-1]
    return None


def _resolve_ok(path: list):
    """Walk a dotted path from an allowed module; return an error if it reaches a module outside torch/math,
    or an object defined in one (for example torch.os or torch.nn.functional.warnings)."""
    import importlib
    try:
        obj = importlib.import_module(path[0])
    except ImportError:
        return None
    for i, name in enumerate(path[1:], start=1):
        if not hasattr(obj, name):
            return None  # attribute of a runtime value or a typo: caught when the file runs
        obj = getattr(obj, name)
        if isinstance(obj, types.ModuleType):
            mod = obj.__name__
        else:
            mod = getattr(obj, "__module__", None) or ""
            if not isinstance(mod, str):
                mod = ""
        top = mod.split(".")[0]
        if isinstance(obj, types.ModuleType) and top not in ("torch", "math"):
            return f"'{'.'.join(path[:i + 1])}' reaches module '{mod}', which is not allowed"
        if top in ("os", "sys", "subprocess", "builtins", "importlib", "shutil", "socket", "pickle", "ctypes",
                   "inspect", "io", "pathlib", "threading", "multiprocessing", "signal", "tempfile", "glob", "posix",
                   "runpy", "code", "codeop", "marshal", "urllib", "http", "requests", "numpy"):
            return f"'{'.'.join(path[:i + 1])}' comes from module '{mod}', which is not allowed"
    return None


def static_check(src: str) -> list:
    errs = []
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"syntax error line {e.lineno}: {e.msg}"]
    aliases = {}  # local name -> module path
    for node in tree.body:
        ok = isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            ok = True  # docstring
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            val = node.value
            try:
                ast.literal_eval(val) if val is not None else None
                ok = True
            except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                ok = False
        if not ok:
            errs.append(f"line {node.lineno}: top-level code may only import, define functions/classes and assign constants")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name not in ALLOWED_IMPORTS:
                    errs.append(f"line {node.lineno}: import of '{a.name}' is not allowed")
                aliases[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for a in node.names:
                full = f"{mod}.{a.name}"
                if full not in ALLOWED_IMPORTS and mod not in ("torch.nn", "torch.nn.functional", "math", "torch"):
                    errs.append(f"line {node.lineno}: 'from {mod} import {a.name}' is not allowed")
                elif mod == "torch" and a.name not in ("nn", "Tensor"):
                    errs.append(f"line {node.lineno}: 'from torch import {a.name}' is not allowed; use torch.{a.name}")
                elif mod in ("torch.nn", "torch.nn.functional") and (a.name in BANNED_TORCH or a.name in ("utils", "parallel")):
                    errs.append(f"line {node.lineno}: 'from {mod} import {a.name}' is not allowed")
                elif a.name == "*":
                    errs.append(f"line {node.lineno}: star imports are not allowed")
                elif mod in ("torch", "torch.nn", "torch.nn.functional", "math"):
                    e = _resolve_ok(mod.split(".") + [a.name])
                    if e:
                        errs.append(f"line {node.lineno}: {e}")
                aliases[a.asname or a.name] = full
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            errs.append(f"line {node.lineno}: '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_") and node.attr not in SAFE_DUNDERS:
                errs.append(f"line {node.lineno}: attribute '{node.attr}' is not allowed (leading underscore)")
            if node.attr in BANNED_ATTRS:
                errs.append(f"line {node.lineno}: '.{node.attr}' is not allowed")
            chain = _torch_chain(node)
            if chain:
                root = aliases.get(chain[0], chain[0])
                path = root.split(".") + chain[1:]
                if path[0] == "torch" and len(path) > 1:
                    sub = path[1] if path[1] not in ("nn",) else None
                    if sub in BANNED_TORCH:
                        errs.append(f"line {node.lineno}: 'torch.{sub}' is not allowed")
                    if len(path) > 2 and path[1] == "nn" and path[2] in ("utils", "parallel"):
                        errs.append(f"line {node.lineno}: 'torch.nn.{path[2]}' is not allowed")
                if root.split(".")[0] in ("torch", "math") and isinstance(node.ctx, (ast.Store, ast.Del)):
                    errs.append(f"line {node.lineno}: changing library attributes is not allowed")
                if root.split(".")[0] in ("torch", "math") and chain[0] in aliases:
                    e = _resolve_ok(path)
                    if e:
                        errs.append(f"line {node.lineno}: {e}")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            errs.append(f"line {node.lineno}: global/nonlocal is not allowed")
        elif isinstance(node, (ast.AsyncFunctionDef, ast.Await, ast.AsyncFor, ast.AsyncWith, ast.Yield, ast.YieldFrom)):
            errs.append(f"line {node.lineno}: async code and generators are not allowed")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) > 2000:
            errs.append(f"line {node.lineno}: string constant too long")
    if not any(isinstance(n, ast.FunctionDef) and n.name == "build_policy" for n in tree.body):
        errs.append("no top-level function 'build_policy(spec)'")
    if len(src) > 40_000:
        errs.append("file longer than 40,000 characters")
    return sorted(set(errs), key=lambda s: (len(s), s))


def _safe_builtins():
    import builtins
    names = ("abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "isinstance", "issubclass", "len",
             "list", "max", "min", "range", "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple",
             "zip", "map", "filter", "super", "ValueError", "RuntimeError", "AssertionError", "NotImplementedError",
             "TypeError", "object", "pow", "divmod", "frozenset", "hasattr", "__build_class__", "Exception",
             "staticmethod", "classmethod", "property", "callable", "iter", "next", "id", "hash", "repr", "format")
    b = {n: getattr(builtins, n) for n in names}
    b["print"] = lambda *a, **k: None
    # torch's C++ code imports its own submodules through the calling frame's builtins, so a restricted
    # __import__ breaks ordinary calls such as torch.tensor. Import statements are limited by the static
    # check instead, and the name __import__ itself is banned there.
    b["__import__"] = builtins.__import__
    return b


def load_candidate(src: str):
    """Static-check, then exec the source with restricted builtins; returns build_policy."""
    errs = static_check(src)
    if errs:
        raise CandidateError("; ".join(errs[:12]))
    mod = types.ModuleType("candidate")
    mod.__dict__["__builtins__"] = _safe_builtins()
    mod.__dict__["__name__"] = "candidate"
    exec(compile(src, "candidate.py", "exec"), mod.__dict__)  # noqa: S102 - checked above, restricted builtins
    fn = mod.__dict__.get("build_policy")
    if not callable(fn):
        raise CandidateError("build_policy is not callable")
    return fn


def sha(src: str) -> str:
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def runtime_check(src: str, spec, max_params: int, max_fb_ms: float, batch: int = 128, reps: int = 3) -> dict:
    """Build the candidate on `spec`, check params, shapes, finiteness and forward+backward time."""
    from arch_rsi.interface import CandidatePolicy
    build = load_candidate(src)
    torch.manual_seed(0)
    try:
        net = build(spec)
    except Exception as e:  # noqa: BLE001
        raise CandidateError(f"build_policy raised {type(e).__name__}: {str(e)[:300]}") from e
    if not isinstance(net, torch.nn.Module):
        raise CandidateError("build_policy must return a torch.nn.Module")
    pol = CandidatePolicy(net, spec)
    n_params = sum(p.numel() for p in pol.parameters())
    if n_params == 0:
        raise CandidateError("the network has no trainable parameters")
    if n_params > max_params:
        raise CandidateError(f"{n_params:,} parameters exceeds the cap of {max_params:,}")
    g = torch.Generator().manual_seed(1)
    obs = (torch.rand(batch, spec.in_channels, spec.Y, spec.X, generator=g) < 0.3).float()
    mask = torch.rand(batch, spec.n_actions, generator=g) < 0.5
    mask[:, -1] = True
    times = []
    for i in range(reps + 1):
        t0 = time.perf_counter()
        try:
            logits, value = pol(obs, mask)
            loss = torch.log_softmax(logits, -1).mean() + value.pow(2).mean()
            loss.backward()
        except Exception as e:  # noqa: BLE001
            raise CandidateError(f"forward/backward raised {type(e).__name__}: {str(e)[:300]}") from e
        if i:
            times.append((time.perf_counter() - t0) * 1000)
        pol.zero_grad(set_to_none=True)
    if not (torch.isfinite(logits[mask]).all() and torch.isfinite(value).all()):
        raise CandidateError("outputs contain NaN or inf")
    fb_ms = sorted(times)[len(times) // 2]
    if fb_ms > max_fb_ms:
        raise CandidateError(f"forward+backward took {fb_ms:.0f} ms on a batch of {batch}; the cap is {max_fb_ms:.0f} ms")
    return {"params": n_params, "fb_ms": round(fb_ms, 2)}
