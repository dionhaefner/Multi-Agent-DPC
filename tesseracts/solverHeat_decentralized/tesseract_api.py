import sys
import os
import numpy as np
import jax
import jax.numpy as jnp
from typing import Any
from pydantic import BaseModel, Field, ConfigDict
from jax.flatten_util import ravel_pytree
from tesseract_core.runtime import Array, Differentiable, Float32, ShapeDType

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import solver
from models.policy import DecentralizedControlNet

# --- 0. Setup Model & Flattening Logic ---
POLICY_MODEL = DecentralizedControlNet(features=(64, 64))

# We initialize once to capture the "skeleton" of the ntralized model.
# This is beacuse the weights structure is independent of the 
# number of agents because of the broadcast/fusion architecture.
_DUMMY_PARAMS = POLICY_MODEL.init(
    jax.random.PRNGKey(0), 
    jnp.zeros((100,)), # State
    jnp.zeros((100,)), # Target
    jnp.zeros((1,))    # 1 Agent template
)
_INITIAL_FLAT, _UNFLATTEN_FN = ravel_pytree(_DUMMY_PARAMS)
_PARAM_SIZE = _INITIAL_FLAT.size

# --- 1. Define the Input Schema ---
# Shapes use ... to accept both unbatched (1D) and batched (2D) inputs
# when used with apply_tesseract(..., vectorized=True).
class InputSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    z_init: Differentiable[Array[..., Float32]]
    xi_init: Differentiable[Array[..., Float32]]
    z_target: Differentiable[Array[..., Float32]]
    flat_params: Differentiable[Array[..., Float32]] = Field(
        description="Flattened Centralized NN weight vector"
    )

    t_steps: int = Field(default=300)

# --- 2. Define the Output Schema ---
class OutputSchema(BaseModel):
    z_trajectory: Differentiable[Array[..., Float32]]
    xi_trajectory: Differentiable[Array[..., Float32]]
    u_trajectory: Differentiable[Array[..., Float32]]
    v_trajectory: Differentiable[Array[..., Float32]]

# --- 3. Apply Function ---
@jax.jit(static_argnames="t_steps")
def _solve_single(z_init, xi_init, z_target, flat_params, t_steps):
    """Solve for a single sample."""
    params = _UNFLATTEN_FN(flat_params)
    return solver.solve_with_policy(
        z_init, xi_init, z_target, params, POLICY_MODEL.apply, t_steps
    )

@jax.jit(static_argnames="t_steps")
def _solve_batch(z_init, xi_init, z_target, flat_params, t_steps):
    """Solve for a batch of samples."""
    solve_fn = lambda z, xi, zt, fp: _solve_single(z, xi, zt, fp, t_steps)
    return jax.vmap(solve_fn)(z_init, xi_init, z_target, flat_params)


def apply(inputs: InputSchema) -> OutputSchema:
    ndim = inputs.z_init.ndim
    if ndim == 1:
        # Single sample
        z_traj, xi_traj, u_traj, v_traj = _solve_single(
            inputs.z_init, inputs.xi_init, inputs.z_target,
            inputs.flat_params, t_steps=inputs.t_steps,
        )
    elif ndim == 2:
        # Batched (leading batch dim from vectorized=True)
        z_traj, xi_traj, u_traj, v_traj = _solve_batch(
            inputs.z_init, inputs.xi_init, inputs.z_target, inputs.flat_params,
            t_steps=inputs.t_steps,
        )
    else:
        raise ValueError(f"Expected 1D or 2D z_init, got ndim={ndim}")

    return OutputSchema(
        z_trajectory=np.asarray(z_traj),
        xi_trajectory=np.asarray(xi_traj),
        u_trajectory=np.asarray(u_traj),
        v_trajectory=np.asarray(v_traj),
    )

# --- 4. Abstract Evaluation (Dynamic Agent Support) ---
def abstract_eval(abstract_inputs: InputSchema):
    t_steps = abstract_inputs.t_steps
    shape = abstract_inputs.z_init.shape

    if len(shape) == 1:
        z_dim = shape[0]
        n_agents = abstract_inputs.xi_init.shape[0]
        prefix = ()
    else:
        batch, z_dim = shape[0], shape[1]
        n_agents = abstract_inputs.xi_init.shape[1]
        prefix = (batch,)

    return {
        "z_trajectory": ShapeDType(shape=(*prefix, t_steps, z_dim), dtype="float32"),
        "xi_trajectory": ShapeDType(shape=(*prefix, t_steps, n_agents), dtype="float32"),
        "u_trajectory": ShapeDType(shape=(*prefix, t_steps, n_agents), dtype="float32"),
        "v_trajectory": ShapeDType(shape=(*prefix, t_steps, n_agents), dtype="float32"),
    }

# --- 5. VJP Function ---

_INPUT_NAMES = ("z_init", "xi_init", "z_target", "flat_params")


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    ndim = inputs.z_init.ndim

    def forward(z_i, xi_i, z_t, p_flat):
        if ndim <= 1:
            return _solve_single(z_i, xi_i, z_t, p_flat, inputs.t_steps)
        else:
            solve_batch = jax.vmap(
                lambda z, xi, zt, fp: _solve_single(z, xi, zt, fp, inputs.t_steps)
            )
            return solve_batch(z_i, xi_i, z_t, p_flat)

    primal_out, vjp_fn = jax.vjp(
        forward,
        inputs.z_init,
        inputs.xi_init,
        inputs.z_target,
        inputs.flat_params,
    )

    cotan_tuple = (
        cotangent_vector.get("z_trajectory"),
        cotangent_vector.get("xi_trajectory"),
        cotangent_vector.get("u_trajectory"),
        cotangent_vector.get("v_trajectory"),
    )

    grads = vjp_fn(cotan_tuple)

    all_grads = dict(zip(_INPUT_NAMES, grads))
    return {k: all_grads[k] for k in vjp_inputs if k in all_grads}