"""No integer seed literal outside sgs/config.py."""

import ast
import re
from pathlib import Path

import pytest

import sgs

SGS_DIR = Path(sgs.__file__).parent
MODULES = sorted(p for p in SGS_DIR.glob("*.py") if p.name != "config.py")
SEED_NAME = re.compile(r"seed|random_state", re.IGNORECASE)
RNG_CALLS = {"LatinHypercube", "default_rng", "RandomState", "check_random_state"}


def _int_literal(node):
    return (isinstance(node, ast.Constant) and isinstance(node.value, int)
            and not isinstance(node.value, bool))


def _target_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
        return str(node.slice.value)
    return None


def _call_name(node):
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def seed_violations(source):
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if node.value is not None and _int_literal(node.value):
                for target in targets:
                    name = _target_name(target)
                    if name and SEED_NAME.search(name):
                        found.append(f"line {node.lineno}: {name} = literal")
        elif isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg and SEED_NAME.search(kw.arg) and _int_literal(kw.value):
                    found.append(f"line {node.lineno}: {kw.arg}=literal")
            if _call_name(node) in RNG_CALLS and any(_int_literal(a) for a in node.args):
                found.append(f"line {node.lineno}: {_call_name(node)}(literal)")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args.args + node.args.kwonlyargs
            defaults = ([None] * (len(node.args.args) - len(node.args.defaults))
                        + node.args.defaults + node.args.kw_defaults)
            for arg, default in zip(args, defaults):
                if default is not None and SEED_NAME.search(arg.arg) and _int_literal(default):
                    found.append(f"line {node.lineno}: default {arg.arg}=literal")
    return found


@pytest.mark.parametrize("snippet", [
    "seed = 42",
    "RANDOM_STATE: int = 0",
    "f(random_state=0)",
    "qmc.LatinHypercube(42)",
    "np.random.default_rng(7)",
    "def f(seed=1): pass",
    "cfg['seed'] = 3",
])
def test_checker_catches(snippet):
    assert seed_violations(snippet)


@pytest.mark.parametrize("snippet", [
    "seed = config.SEEDS['seed_lhs_design']",
    "f(random_state=config.SEEDS['protocol_estimator'], n_jobs=1)",
    "qmc.LatinHypercube(d=len(config.PARAMS), seed=seed)",
    "np.random.default_rng(config.SEEDS['guided_stream'])",
    "n_seed = len(frame)",
])
def test_checker_allows(snippet):
    assert not seed_violations(snippet)


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_no_seed_literal_outside_config(path):
    assert seed_violations(path.read_text()) == []
