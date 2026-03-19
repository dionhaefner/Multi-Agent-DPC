"""
Minimal reproduction of the performance difference between:
  1. use_tesseract=True  + Tesseract.from_image        (~4s/iter)
  2. use_tesseract=True  + Tesseract.from_tesseract_api (~1s/iter)
  3. use_tesseract=False  (native JAX)                  (~0.3s/iter)

This script isolates a single jitted, vmapped call through
apply_tesseract (or the raw solver) and times it.
Use --no-grad to run forward-only and see how much of the overhead
comes from the backward pass vs the forward pass.

Usage:
  python repro_perf.py --mode all              # fwd+bwd, all backends
  python repro_perf.py --mode all --no-grad    # fwd only, all backends
  python repro_perf.py --mode from_api
  python repro_perf.py --mode from_api --tesseract noop   # framework overhead only
  python repro_perf.py --mode native --no-grad
"""
import argparse
import time
import sys
from pathlib import Path
from functools import partial

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

# --- project setup ---
script_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(script_dir))

from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract
from tesseracts.solverHeat_decentralized import solver
from models.policy import DecentralizedControlNet

# ---------- constants (same as train.py) ----------
n_pde, n_agents, batch_size = 100, 8, 32  # same as train.py
T_steps = 300

# Tesseract API paths
TESSERACT_API_PATHS = {
    "solver": script_dir / "tesseracts/solverHeat_decentralized/tesseract_api.py",
    "noop": script_dir / "tesseracts/noop/tesseract_api.py",
}

# ---------- model + params ----------
model = DecentralizedControlNet(features=(64, 64))
key = jax.random.PRNGKey(0)
params = model.init(key, jnp.zeros((n_pde,)), jnp.zeros((n_pde,)), jnp.zeros((n_agents,)))


# ============================================================
# Three forward+backward paths
# ============================================================
def loss_via_tesseract(params, z_init, xi_init, z_target, solver_ts, vectorized=False):
    """Forward+backward through apply_tesseract."""
    flat_params, _ = ravel_pytree(params)
    inputs = {
        "z_init": z_init,
        "xi_init": xi_init,
        "z_target": z_target,
        "flat_params": flat_params,
        "t_steps": T_steps,
    }
    results = apply_tesseract(solver_ts, inputs, vectorized=vectorized)
    z_traj = results["z_trajectory"]
    return jnp.mean((z_traj - z_target[None, :]) ** 2)


def loss_native(params, z_init, xi_init, z_target):
    """Forward+backward through raw JAX solver (no tesseract)."""
    z_traj, _, _, _ = solver.solve_with_policy(
        z_init, xi_init, z_target, params, model.apply, T_steps
    )
    return jnp.mean((z_traj - z_target[None, :]) ** 2)


def build_step_tesseract(solver_ts, use_grad, vectorized=False):
    """Build a jitted, vmapped step going through apply_tesseract."""
    @partial(jax.jit, static_argnames="solver_ts")
    def step_grad(params, z_init_b, xi_init_b, z_target_b, solver_ts):
        batched = jax.vmap(
            lambda p, z, xi, zt, ts: loss_via_tesseract(p, z, xi, zt, ts, vectorized=vectorized),
            in_axes=(None, 0, 0, 0, None),
        )
        def mean_loss(p):
            return jnp.mean(batched(p, z_init_b, xi_init_b, z_target_b, solver_ts))
        loss, grads = jax.value_and_grad(mean_loss)(params)
        return loss, grads

    @partial(jax.jit, static_argnames="solver_ts")
    def step_fwd(params, z_init_b, xi_init_b, z_target_b, solver_ts):
        batched = jax.vmap(
            lambda p, z, xi, zt, ts: loss_via_tesseract(p, z, xi, zt, ts, vectorized=vectorized),
            in_axes=(None, 0, 0, 0, None),
        )
        loss = jnp.mean(batched(params, z_init_b, xi_init_b, z_target_b, solver_ts))
        return loss, None

    return step_grad if use_grad else step_fwd


@jax.jit
def step_native_grad(params, z_init_b, xi_init_b, z_target_b):
    """Jitted, vmapped value_and_grad through raw JAX solver."""
    batched = jax.vmap(loss_native, in_axes=(None, 0, 0, 0))
    def mean_loss(p):
        return jnp.mean(batched(p, z_init_b, xi_init_b, z_target_b))
    loss, grads = jax.value_and_grad(mean_loss)(params)
    return loss, grads


@jax.jit
def step_native_fwd(params, z_init_b, xi_init_b, z_target_b):
    """Jitted, vmapped forward-only through raw JAX solver."""
    batched = jax.vmap(loss_native, in_axes=(None, 0, 0, 0))
    loss = jnp.mean(batched(params, z_init_b, xi_init_b, z_target_b))
    return loss, None


# ============================================================
# Raw forward-pass benchmarks (no JAX transformations at all)
# ============================================================
flat_params, _unflatten = ravel_pytree(params)

def _make_single_input():
    """Single (non-batched) input for raw .apply() calls."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(42))
    return {
        "z_init": jax.random.normal(k1, (n_pde,)),
        "xi_init": jnp.linspace(0.2, 0.8, n_agents),
        "z_target": jax.random.normal(k2, (n_pde,)),
        "flat_params": flat_params,
        "t_steps": T_steps,
    }


def run_raw_from_image():
    print("=== Raw forward: Tesseract.from_image (Docker) ===")
    solver_ts = Tesseract.from_image("solver_heat_decentralized:latest")
    inputs = _make_single_input()
    with solver_ts:
        bench_raw(lambda: solver_ts.apply(inputs))


def run_raw_from_api(api_path=None):
    api_path = api_path or TESSERACT_API_PATHS["solver"]
    print(f"=== Raw forward: Tesseract.from_tesseract_api (LocalClient) [{api_path.name}] ===")
    solver_ts = Tesseract.from_tesseract_api(api_path)
    inputs = _make_single_input()
    with solver_ts:
        bench_raw(lambda: solver_ts.apply(inputs))


def _make_array_inputs():
    """Differentiable array inputs only (t_steps kept as Python int)."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(42))
    return {
        "z_init": jax.random.normal(k1, (n_pde,)),
        "xi_init": jnp.linspace(0.2, 0.8, n_agents),
        "z_target": jax.random.normal(k2, (n_pde,)),
        "flat_params": flat_params,
    }


def run_raw_apply_tesseract_from_image():
    print("=== Raw forward: jit(apply_tesseract) + from_image (Docker) ===")
    solver_ts = Tesseract.from_image("solver_heat_decentralized:latest")
    array_inputs = _make_array_inputs()
    with solver_ts:
        @jax.jit
        def call(array_inputs):
            inputs = {**array_inputs, "t_steps": T_steps}
            return apply_tesseract(solver_ts, inputs)
        bench_raw(lambda: call(array_inputs))


def run_raw_apply_tesseract_from_api(api_path=None):
    api_path = api_path or TESSERACT_API_PATHS["solver"]
    print(f"=== Raw forward: jit(apply_tesseract) + from_api (LocalClient) [{api_path.name}] ===")
    solver_ts = Tesseract.from_tesseract_api(api_path)
    array_inputs = _make_array_inputs()
    with solver_ts:
        @jax.jit
        def call(array_inputs):
            inputs = {**array_inputs, "t_steps": T_steps}
            return apply_tesseract(solver_ts, inputs)
        bench_raw(lambda: call(array_inputs))


def run_raw_native():
    print("=== Raw forward: solver.solve_with_policy (direct call) ===")
    inputs = _make_single_input()
    # Pre-jit the solver so we only measure execution, not tracing
    jitted_solve = jax.jit(
        solver.solve_with_policy, static_argnums=(4, 5)
    )
    def call():
        return jitted_solve(
            inputs["z_init"], inputs["xi_init"], inputs["z_target"],
            params, model.apply, T_steps,
        )
    bench_raw(call)


# ============================================================
# Dissect: isolate each layer of the VJP backward pass
# ============================================================
def run_dissect(api_path=None):
    """Benchmark each layer of the backward pass independently (single sample)."""
    api_path = api_path or TESSERACT_API_PATHS["solver"]
    from tesseracts.solverHeat_decentralized import tesseract_api
    from tesseract_jax import apply_tesseract

    inputs = _make_single_input()
    array_inputs = _make_array_inputs()
    z_init = inputs["z_init"]
    xi_init = inputs["xi_init"]
    z_target = inputs["z_target"]

    # --- A. Native jax.value_and_grad (single sample, the baseline) ---
    print("=== [A] Native jax.value_and_grad (single sample, jitted) ===")
    @jax.jit
    def native_val_grad(p, z_i, xi_i, z_t):
        def fwd(p):
            z_traj, _, _, _ = solver.solve_with_policy(
                z_i, xi_i, z_t, p, model.apply, T_steps
            )
            return jnp.mean((z_traj - z_t[None, :]) ** 2)
        return jax.value_and_grad(fwd)(p)
    bench_raw(lambda: native_val_grad(params, z_init, xi_init, z_target))

    # --- B. tesseract_api.vector_jacobian_product called directly ---
    #     (bypasses Tesseract SDK, Jaxeract, Pydantic — just the user's jax.vjp code)
    print("=== [B] tesseract_api.vector_jacobian_product (direct call, no SDK) ===")
    # First do a forward pass to get realistic cotangent shapes
    fwd_out = tesseract_api.apply(tesseract_api.InputSchema(**inputs))
    cotan = {
        "z_trajectory": jnp.ones_like(fwd_out.z_trajectory),
        "xi_trajectory": jnp.ones_like(fwd_out.xi_trajectory),
        "u_trajectory": jnp.ones_like(fwd_out.u_trajectory),
        "v_trajectory": jnp.ones_like(fwd_out.v_trajectory),
    }
    def call_vjp_direct():
        return tesseract_api.vector_jacobian_product(
            tesseract_api.InputSchema(**inputs),
            vjp_inputs={"z_init", "xi_init", "z_target", "flat_params"},
            vjp_outputs={"z_trajectory", "xi_trajectory", "u_trajectory", "v_trajectory"},
            cotangent_vector=cotan,
        )
    bench_raw(call_vjp_direct)

    # --- C. solver_ts.vector_jacobian_product (through LocalClient + Pydantic) ---
    print("=== [C] solver_ts.vector_jacobian_product (LocalClient + Pydantic) ===")
    solver_ts = Tesseract.from_tesseract_api(api_path)
    with solver_ts:
        def call_vjp_sdk():
            return solver_ts.vector_jacobian_product(
                inputs=inputs,
                vjp_inputs=["z_init", "xi_init", "z_target", "flat_params"],
                vjp_outputs=["z_trajectory", "xi_trajectory", "u_trajectory", "v_trajectory"],
                cotangent_vector=cotan,
            )
        bench_raw(call_vjp_sdk)

    # D skipped: single-sample grad through apply_tesseract where only
    # flat_params has tangents triggers a bug in the Jaxeract VJP output
    # reconstruction loop (returns 3 outputs instead of 4). Not relevant
    # to the vmap performance investigation.

    # --- E. jit(vmap(value_and_grad(apply_tesseract))) batch=4 ---
    #     (full training path, same as --mode from_api)
    print(f"=== [E] jit(vmap(value_and_grad(apply_tesseract))) batch={batch_size} ===")
    solver_ts3 = Tesseract.from_tesseract_api(api_path)
    with solver_ts3:
        step = build_step_tesseract(solver_ts3, use_grad=True)
        data = make_dummy_data(jax.random.PRNGKey(42))
        bench_raw(lambda: step(data[0], data[1], data[2], data[3], solver_ts3))

    # --- F. Native jax.value_and_grad, vmapped, batch=4 ---
    #     (to see if native vmap also scales linearly)
    print(f"=== [F] Native jit(vmap(value_and_grad)) batch={batch_size} ===")
    bench_raw(lambda: step_native_grad(*data))

    # --- G. jit(vmap(value_and_grad(apply_tesseract))) batch=4 with vectorized=True ---
    #     (single callback per batch instead of one per sample)
    print(f"=== [G] jit(vmap(value_and_grad(apply_tesseract))) batch={batch_size}, vectorized=True ===")
    solver_ts4 = Tesseract.from_tesseract_api(api_path)
    with solver_ts4:
        @partial(jax.jit, static_argnames="solver_ts")
        def step_vectorized(params, z_init_b, xi_init_b, z_target_b, solver_ts):
            batched = jax.vmap(
                lambda p, z, xi, zt, ts: loss_via_tesseract(p, z, xi, zt, ts, vectorized=True),
                in_axes=(None, 0, 0, 0, None),
            )
            def mean_loss(p):
                return jnp.mean(batched(p, z_init_b, xi_init_b, z_target_b, solver_ts))
            loss, grads = jax.value_and_grad(mean_loss)(params)
            return loss, grads
        bench_raw(lambda: step_vectorized(data[0], data[1], data[2], data[3], solver_ts4))

    # --- H. jit(vmap(grad(apply_tesseract))) batch=N, vectorized=True (grad only, no value) ---
    print(f"=== [H] jit(vmap(grad(apply_tesseract))) batch={batch_size}, vectorized=True (grad only) ===")
    solver_ts5 = Tesseract.from_tesseract_api(api_path)
    with solver_ts5:
        @partial(jax.jit, static_argnames="solver_ts")
        def step_grad_only(params, z_init_b, xi_init_b, z_target_b, solver_ts):
            batched = jax.vmap(
                lambda p, z, xi, zt, ts: loss_via_tesseract(p, z, xi, zt, ts, vectorized=True),
                in_axes=(None, 0, 0, 0, None),
            )
            def mean_loss(p):
                return jnp.mean(batched(p, z_init_b, xi_init_b, z_target_b, solver_ts))
            grads = jax.grad(mean_loss)(params)
            return grads
        bench_raw(lambda: step_grad_only(data[0], data[1], data[2], data[3], solver_ts5))

    # --- I. Native jit(vmap(grad)) batch=N (grad only, no value) ---
    print(f"=== [I] Native jit(vmap(grad)) batch={batch_size} (grad only) ===")
    @jax.jit
    def step_native_grad_only(params, z_init_b, xi_init_b, z_target_b):
        batched = jax.vmap(loss_native, in_axes=(None, 0, 0, 0))
        def mean_loss(p):
            return jnp.mean(batched(p, z_init_b, xi_init_b, z_target_b))
        return jax.grad(mean_loss)(params)
    bench_raw(lambda: step_native_grad_only(*data))


def run_batch_sweep(api_path=None):
    """Sweep batch sizes to see how the Tesseract vs native gap scales."""
    api_path = api_path or TESSERACT_API_PATHS["solver"]
    sizes = [1, 2, 4, 8, 16, 32]
    n_iter = 5

    print(f"{'batch':>5}  {'G (vectorized)':>14}  {'N (noop tess)':>14}  {'F (native)':>14}  {'G/F':>5}  {'N/F':>5}")
    print("-" * 78)

    for bs in sizes:
        k1, k2 = jax.random.split(jax.random.PRNGKey(42))
        z_b = jax.random.normal(k1, (bs, n_pde))
        zt_b = jax.random.normal(k2, (bs, n_pde))
        xi_b = jnp.tile(jnp.linspace(0.2, 0.8, n_agents), (bs, 1))

        # G: vectorized tesseract (real compute)
        solver_ts = Tesseract.from_tesseract_api(api_path)
        with solver_ts:
            @partial(jax.jit, static_argnames="solver_ts")
            def step_g(params, z_b, xi_b, zt_b, solver_ts):
                batched = jax.vmap(
                    lambda p, z, xi, zt, ts: loss_via_tesseract(p, z, xi, zt, ts, vectorized=True),
                    in_axes=(None, 0, 0, 0, None),
                )
                def mean_loss(p):
                    return jnp.mean(batched(p, z_b, xi_b, zt_b, solver_ts))
                return jax.value_and_grad(mean_loss)(params)

            # warmup
            jax.block_until_ready(step_g(params, z_b, xi_b, zt_b, solver_ts))
            times_g = []
            for _ in range(n_iter):
                t0 = time.perf_counter()
                jax.block_until_ready(step_g(params, z_b, xi_b, zt_b, solver_ts))
                times_g.append(time.perf_counter() - t0)

        # N: no-op tesseract (framework overhead only)
        noop_ts = Tesseract.from_tesseract_api(TESSERACT_API_PATHS["noop"])
        with noop_ts:
            @partial(jax.jit, static_argnames="solver_ts")
            def step_n(params, z_b, xi_b, zt_b, solver_ts):
                batched = jax.vmap(
                    lambda p, z, xi, zt, ts: loss_via_tesseract(p, z, xi, zt, ts, vectorized=True),
                    in_axes=(None, 0, 0, 0, None),
                )
                def mean_loss(p):
                    return jnp.mean(batched(p, z_b, xi_b, zt_b, solver_ts))
                return jax.value_and_grad(mean_loss)(params)

            jax.block_until_ready(step_n(params, z_b, xi_b, zt_b, noop_ts))
            times_n = []
            for _ in range(n_iter):
                t0 = time.perf_counter()
                jax.block_until_ready(step_n(params, z_b, xi_b, zt_b, noop_ts))
                times_n.append(time.perf_counter() - t0)

        # F: native
        @jax.jit
        def step_f(params, z_b, xi_b, zt_b):
            batched = jax.vmap(loss_native, in_axes=(None, 0, 0, 0))
            def mean_loss(p):
                return jnp.mean(batched(p, z_b, xi_b, zt_b))
            return jax.value_and_grad(mean_loss)(params)

        jax.block_until_ready(step_f(params, z_b, xi_b, zt_b))
        times_f = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            jax.block_until_ready(step_f(params, z_b, xi_b, zt_b))
            times_f.append(time.perf_counter() - t0)

        avg_g = sum(times_g) / n_iter
        avg_n = sum(times_n) / n_iter
        avg_f = sum(times_f) / n_iter
        print(f"{bs:>5}  {avg_g:>13.3f}s  {avg_n:>13.3f}s  {avg_f:>13.3f}s  {avg_g/avg_f:>4.1f}x  {avg_n/avg_f:>4.1f}x")


# ============================================================
# Timing harness
# ============================================================
def make_dummy_data(key):
    k1, k2 = jax.random.split(key)
    z_init_b = jax.random.normal(k1, (batch_size, n_pde))
    z_target_b = jax.random.normal(k2, (batch_size, n_pde))
    xi_init_b = jnp.tile(jnp.linspace(0.2, 0.8, n_agents), (batch_size, 1))
    return params, z_init_b, xi_init_b, z_target_b


def bench_raw(call_fn, n_warmup=1, n_iter=5):
    """Time a zero-arg callable (raw forward pass)."""
    print(f"  warmup ({n_warmup} iters)...")
    for i in range(n_warmup):
        t0 = time.perf_counter()
        out = call_fn()
        jax.block_until_ready(out)
        dt = time.perf_counter() - t0
        print(f"    warmup {i}: {dt:.3f}s")

    print(f"  timed ({n_iter} iters)...")
    times = []
    for i in range(n_iter):
        t0 = time.perf_counter()
        out = call_fn()
        jax.block_until_ready(out)
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"    iter {i}: {dt:.3f}s")

    avg = sum(times) / len(times)
    print(f"  => average: {avg:.3f}s/iter\n")


def bench(step_fn, n_warmup=1, n_iter=3):
    """Run step_fn, printing per-iteration wall time."""
    data = make_dummy_data(jax.random.PRNGKey(42))

    # warmup (includes compilation)
    print(f"  warmup ({n_warmup} iters, includes jit compilation)...")
    for i in range(n_warmup):
        t0 = time.perf_counter()
        loss, grads = step_fn(*data)
        jax.block_until_ready((loss, grads))
        dt = time.perf_counter() - t0
        print(f"    warmup {i}: {dt:.2f}s  loss={float(loss):.6f}")

    # timed
    print(f"  timed ({n_iter} iters)...")
    times = []
    for i in range(n_iter):
        t0 = time.perf_counter()
        loss, grads = step_fn(*data)
        jax.block_until_ready((loss, grads))
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"    iter {i}: {dt:.2f}s  loss={float(loss):.6f}")

    avg = sum(times) / len(times)
    print(f"  => average: {avg:.2f}s/iter\n")


# ============================================================
# Main
# ============================================================
def run_from_image(use_grad, vectorized=False):
    vtag = ", vectorized" if vectorized else ""
    tag = "fwd+bwd" if use_grad else "fwd only"
    print(f"=== Mode: Tesseract.from_image (Docker) [{tag}{vtag}] ===")
    if vectorized:
        print("  NOTE: Docker image must be rebuilt with Array[..., Float32] schemas")
        print("  Run: tesseract build tesseracts/solverHeat_decentralized -t solver_heat_decentralized:latest")
    solver_ts = Tesseract.from_image("solver_heat_decentralized:latest")
    with solver_ts:
        step = build_step_tesseract(solver_ts, use_grad, vectorized=vectorized)
        step_fn = lambda p, z, xi, zt: step(p, z, xi, zt, solver_ts)
        bench(step_fn)


def run_from_api(use_grad, vectorized=False, api_path=None):
    api_path = api_path or TESSERACT_API_PATHS["solver"]
    vtag = ", vectorized" if vectorized else ""
    tag = "fwd+bwd" if use_grad else "fwd only"
    print(f"=== Mode: Tesseract.from_tesseract_api (LocalClient) [{tag}{vtag}, {api_path.name}] ===")
    solver_ts = Tesseract.from_tesseract_api(api_path)
    with solver_ts:
        step = build_step_tesseract(solver_ts, use_grad, vectorized=vectorized)
        step_fn = lambda p, z, xi, zt: step(p, z, xi, zt, solver_ts)
        bench(step_fn)


def run_native(use_grad, vectorized=False):
    tag = "fwd+bwd" if use_grad else "fwd only"
    print(f"=== Mode: Native JAX (no tesseract) [{tag}] ===")
    step = step_native_grad if use_grad else step_native_fwd
    step_fn = lambda p, z, xi, zt: step(p, z, xi, zt)
    bench(step_fn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["from_image", "from_api", "native", "all", "raw", "dissect", "sweep"],
        required=True,
        help="Which code path to benchmark",
    )
    parser.add_argument(
        "--no-grad",
        action="store_true",
        help="Forward pass only (no value_and_grad) — only applies to non-raw modes",
    )
    parser.add_argument(
        "--vectorized",
        action="store_true",
        help="Use vectorized=True for apply_tesseract (single callback per batch)",
    )
    parser.add_argument(
        "--tesseract",
        choices=["solver", "noop"],
        default="solver",
        help="Which tesseract API to use: 'solver' (real compute) or 'noop' (framework overhead only)",
    )
    args = parser.parse_args()
    use_grad = not args.no_grad
    api_path = TESSERACT_API_PATHS[args.tesseract]

    grad_modes = {
        "from_image": run_from_image,
        "from_api": run_from_api,
        "native": run_native,
    }

    raw_modes = {
        "from_image": run_raw_from_image,
        "from_api": run_raw_from_api,
        "apply_tesseract+from_image": run_raw_apply_tesseract_from_image,
        "apply_tesseract+from_api": run_raw_apply_tesseract_from_api,
        "native": run_raw_native,
    }

    if args.mode == "dissect":
        run_dissect(api_path)
        return

    if args.mode == "sweep":
        run_batch_sweep(api_path)
        return

    if args.mode == "raw":
        for name, run in raw_modes.items():
            try:
                if "from_api" in name:
                    run(api_path=api_path)
                else:
                    run()
            except Exception as e:
                print(f"  skipping {name}: {e}\n")
    elif args.mode == "all":
        for name, run in grad_modes.items():
            try:
                if name == "from_api":
                    run(use_grad, vectorized=args.vectorized, api_path=api_path)
                else:
                    run(use_grad, vectorized=args.vectorized)
            except Exception as e:
                print(f"  skipping {name}: {e}\n")
    else:
        if args.mode == "from_api":
            grad_modes[args.mode](use_grad, vectorized=args.vectorized, api_path=api_path)
        else:
            grad_modes[args.mode](use_grad, vectorized=args.vectorized)


if __name__ == "__main__":
    main()
