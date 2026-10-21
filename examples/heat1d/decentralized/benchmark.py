"""
Times one training step of train.py with the solver called in five different ways.

    python benchmark.py                  # all setups, 3 fresh processes each
    python benchmark.py --processes 5 --steps 100

Each setup runs in its own fresh processes, which first run --warmup steps and then
time --steps more. The table reports the median over processes of each process's
median step time, along with the spread across processes. The loss after the last
step is printed as a check that all setups train the same way.

Setups:
  direct          solver imported in this process (use_tesseract=False)
  host            solver served by Tesseract.from_source, arrays through host memory,
                  VJP cache off
  host+cache      same, with the Tesseract's VJP cache (its default)
  cuda_ipc        same as host, arrays passed as CUDA IPC handles, VJP cache off
  cuda_ipc+cache  same, with the Tesseract's VJP cache (its default)
"""
import os

# The training loop and the solver share one GPU, so neither JAX process should
# claim 75% of its memory at startup (from_source passes this on to the solver).
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from functools import partial
from importlib.metadata import version
from pathlib import Path

SETUPS = {
    # name: (use_tesseract, gpu_transport, VJP cache size)
    "direct": (False, None, None),
    "host": (True, "none", "0"),
    "host+cache": (True, "none", "1"),
    "cuda_ipc": (True, "cuda_ipc", "0"),
    "cuda_ipc+cache": (True, "cuda_ipc", "1"),
}
LABELS = {
    "direct": "Solver imported directly, one process",
    "host": "Solver in its own process, arrays through host memory",
    "host+cache": "Same, with the solver's VJP cache",
    "cuda_ipc": "Solver in its own process, cuda_ipc",
    "cuda_ipc+cache": "Same, with the solver's VJP cache",
}


def run_one(setup, warmup, steps):
    """Time `steps` training steps of one setup in this process; print a JSON line."""
    use_tesseract, gpu_transport, cache_size = SETUPS[setup]
    if cache_size is not None:
        os.environ["VJP_CACHE_SIZE"] = cache_size  # inherited by the solver process

    import jax
    import jax.numpy as jnp
    import optax
    from tesseract_core import Tesseract

    script_dir = Path(__file__).resolve().parent.parent.parent.parent
    sys.path.append(str(script_dir))
    from data_utils import generate_grf
    from dynamics_dual import PDEDynamics
    from models.policy import DecentralizedControlNet

    # The same problem, loss and optimizer as train.py.
    n_pde, n_agents, batch_size = 100, 8, 32
    T_steps = 300
    R_safe = 0.05

    model = DecentralizedControlNet(features=(64, 64))
    key = jax.random.PRNGKey(0)
    params = model.init(key, jnp.zeros((n_pde,)), jnp.zeros((n_pde,)), jnp.zeros((n_agents,)))
    lr_schedule = optax.exponential_decay(1e-3, 2000, 0.5)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_schedule))
    opt_state = optimizer.init(params)

    def loss_fn(params, z_init, xi_init, z_target, dynamics):
        z_traj, xi_traj, u_traj, v_traj = dynamics.unroll_controlled(
            z_init, xi_init, z_target, params, T_steps
        )
        l_track = jnp.mean((z_traj - z_target[None, :]) ** 2)
        l_effort = jnp.mean(u_traj ** 2) + 0.1 * jnp.mean(v_traj ** 2)
        margin = 0.02
        l_bound = jnp.mean(jnp.maximum(0, margin - xi_traj)**2 +
                           jnp.maximum(0, xi_traj - (1.0 - margin))**2)
        dists = jnp.abs(xi_traj[:, :, None] - xi_traj[:, None, :])
        mask = jnp.eye(n_agents)[None, :, :]
        l_coll = jnp.mean(jnp.maximum(0, R_safe - (dists + mask * 1.0)) ** 2)
        l_accel = jnp.mean(jnp.diff(v_traj, axis=0)**2)
        return 5.0 * l_track + 0.001 * l_effort + 100.0 * l_bound + 1.0 * l_coll + 0.1 * l_accel

    @partial(jax.jit, static_argnames='dynamics')
    def train_step(params, opt_state, z_init_batch, xi_init_batch, z_target_batch, dynamics):
        batched_loss_fn = jax.vmap(loss_fn, in_axes=(None, 0, 0, 0, None))

        def mean_loss_fn(p):
            return jnp.mean(batched_loss_fn(p, z_init_batch, xi_init_batch, z_target_batch, dynamics))

        loss, grads = jax.value_and_grad(mean_loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    all_keys = jax.random.split(key, 5000)
    _, z_init_all = jax.vmap(partial(generate_grf, n_points=n_pde, length_scale=0.2))(all_keys)
    _, z_target_all = jax.vmap(partial(generate_grf, n_points=n_pde, length_scale=0.4))(all_keys)
    xi_init_batch = jnp.tile(jnp.linspace(0.2, 0.8, n_agents), (batch_size, 1))

    def run(dynamics):
        nonlocal params, opt_state
        times = []
        for i in range(warmup + steps):
            idx = jax.random.randint(jax.random.PRNGKey(i), (batch_size,), 0, 5000)
            z_init_b, z_target_b = z_init_all[idx], z_target_all[idx]
            t0 = time.perf_counter()
            params, opt_state, loss = train_step(
                params, opt_state, z_init_b, xi_init_batch, z_target_b, dynamics
            )
            loss = float(loss)  # waits for the step to finish
            times.append(time.perf_counter() - t0)
        return statistics.median(times[warmup:]) * 1e3, loss

    if not use_tesseract:
        median_ms, loss = run(PDEDynamics(None, policy_apply_fn=model.apply, use_tesseract=False))
    else:
        with Tesseract.from_source(
            script_dir / "tesseracts/solverHeat_decentralized/tesseract_api.py",
            python_executable=sys.executable,
            gpu_transport="cuda_ipc",
        ) as solver_ts:
            # Asking for gpu_transport="none" makes every call copy through the host.
            if gpu_transport == "none":
                solver_ts = solver_ts.with_encoding(gpu_transport="none")
            median_ms, loss = run(PDEDynamics(solver_ts, policy_apply_fn=model.apply))
    print(json.dumps({
        "median_ms": median_ms,
        "loss": loss,
        "gpu": jax.devices()[0].device_kind,
        "versions": {p: version(p) for p in ("jax", "tesseract-core", "tesseract-jax")},
    }))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--setups", nargs="+", default=list(SETUPS), choices=list(SETUPS))
    parser.add_argument("--run", choices=list(SETUPS), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.run:
        run_one(args.run, args.warmup, args.steps)
        return

    results = {}
    # Interleave the setups, so slow drift on the machine affects them all alike.
    for _ in range(args.processes):
        for setup in args.setups:
            out = subprocess.run(
                [sys.executable, __file__, "--run", setup,
                 "--warmup", str(args.warmup), "--steps", str(args.steps)],
                capture_output=True, text=True,
            )
            if out.returncode != 0:
                sys.exit(f"{setup} failed:\n{out.stderr}")
            results.setdefault(setup, []).append(json.loads(out.stdout.strip().splitlines()[-1]))

    first = next(iter(results.values()))[0]
    print(f"GPU: {first['gpu']}, Python {platform.python_version()}")
    print(", ".join(f"{p} {v}" for p, v in first["versions"].items()))
    print(f"{args.processes} processes x {args.steps} timed steps ({args.warmup} warm-up)\n")
    print(f"{'Setup':55s} {'ms/step':>8s} {'spread':>7s} {'final loss':>11s}")
    for setup, runs in results.items():
        ms = [r["median_ms"] for r in runs]
        print(f"{LABELS[setup]:55s} {statistics.median(ms):8.1f} {max(ms) - min(ms):7.1f} "
              f"{statistics.median(r['loss'] for r in runs):11.5f}")


if __name__ == "__main__":
    main()
