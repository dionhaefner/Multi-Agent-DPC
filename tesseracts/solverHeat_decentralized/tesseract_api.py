import os
import sys
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Differentiable, Float32
from tesseract_core.runtime.experimental import set_jax_vjp_cache_size
from tesseract_core.runtime.jax_recipes import (
    jax_abstract_eval,
    jax_apply,
    jax_jacobian,
    jax_jvp,
    jax_vjp,
)

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import solver
from models.policy import DecentralizedControlNet

# --- 0. Setup Model & Flattening Logic ---
POLICY_MODEL = DecentralizedControlNet(features=(64, 64))

# The weights structure is independent of the number of agents because of the
# broadcast/fusion architecture, so one agent is enough to capture it.
_DUMMY_PARAMS = POLICY_MODEL.init(
    jax.random.PRNGKey(0),
    jnp.zeros((100,)),  # State
    jnp.zeros((100,)),  # Target
    jnp.zeros((1,)),  # 1 Agent template
)
_INITIAL_FLAT, _UNFLATTEN_FN = ravel_pytree(_DUMMY_PARAMS)
_PARAM_SIZE = _INITIAL_FLAT.size

# Every gradient evaluation through Tesseract-JAX calls apply and then
# vector_jacobian_product on the same inputs. Caching the residuals of the forward
# pass lets the VJP skip rerunning the 300-step rollout. Set VJP_CACHE_SIZE=0 in
# the environment to disable it.
set_jax_vjp_cache_size(int(os.environ.get("VJP_CACHE_SIZE", "1")))


# --- 1. Schemata ---
# Shapes use ... to accept both unbatched (1D) and batched (2D) inputs, so that
# apply_tesseract(..., vmap_method="broadcast_all") sends a whole batch in one call.
class InputSchema(BaseModel):
    z_init: Differentiable[Array[..., Float32]]
    xi_init: Differentiable[Array[..., Float32]]
    z_target: Differentiable[Array[..., Float32]]
    flat_params: Differentiable[Array[..., Float32]] = Field(
        description=f"Flattened policy weights ({_PARAM_SIZE},), optionally batched"
    )
    t_steps: int = Field(default=300)


class OutputSchema(BaseModel):
    z_trajectory: Differentiable[Array[..., Float32]]
    xi_trajectory: Differentiable[Array[..., Float32]]
    u_trajectory: Differentiable[Array[..., Float32]]
    v_trajectory: Differentiable[Array[..., Float32]]


# --- 2. Solver ---
def _solve_single(z_init, xi_init, z_target, flat_params, t_steps):
    """Solve for a single sample."""
    params = _UNFLATTEN_FN(flat_params)
    return solver.solve_with_policy(
        z_init, xi_init, z_target, params, POLICY_MODEL.apply, t_steps
    )


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    args = (
        inputs["z_init"],
        inputs["xi_init"],
        inputs["z_target"],
        inputs["flat_params"],
    )
    t_steps = inputs["t_steps"]

    ndim = args[0].ndim
    if ndim == 1:
        z_traj, xi_traj, u_traj, v_traj = _solve_single(*args, t_steps)
    elif ndim == 2:
        z_traj, xi_traj, u_traj, v_traj = jax.vmap(
            _solve_single, in_axes=(0, 0, 0, 0, None)
        )(*args, t_steps)
    else:
        raise ValueError(f"Expected 1D or 2D z_init, got ndim={ndim}")

    return {
        "z_trajectory": z_traj,
        "xi_trajectory": xi_traj,
        "u_trajectory": u_traj,
        "v_trajectory": v_traj,
    }


# --- 3. Endpoints ---
def apply(inputs: InputSchema) -> OutputSchema:
    return OutputSchema(**jax_apply(apply_jit, inputs))


def jacobian(
    inputs: InputSchema,
    jac_inputs: set[str],
    jac_outputs: set[str],
):
    return jax_jacobian(apply_jit, inputs, jac_inputs, jac_outputs)


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    return jax_jvp(apply_jit, inputs, jvp_inputs, jvp_outputs, tangent_vector)


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    return jax_vjp(apply_jit, inputs, vjp_inputs, vjp_outputs, cotangent_vector)


def abstract_eval(abstract_inputs):
    return jax_abstract_eval(apply_jit, abstract_inputs)
