import sys
import os
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from pydantic import BaseModel, Field, ConfigDict
from jax.flatten_util import ravel_pytree

from tesseract_core.runtime import Array, Differentiable, Float32
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import solver
from models.policy import DecentralizedControlNet

# --- 0. Setup Model & Flattening Logic ---
POLICY_MODEL = DecentralizedControlNet(features=(64, 64))

_DUMMY_PARAMS = POLICY_MODEL.init(
    jax.random.PRNGKey(0),
    jnp.zeros((100,)),  # State
    jnp.zeros((100,)),  # Target
    jnp.zeros((1,)),    # 1 Agent template
)
_INITIAL_FLAT, _UNFLATTEN_FN = ravel_pytree(_DUMMY_PARAMS)
_PARAM_SIZE = _INITIAL_FLAT.size

#
# Schemata
#

# Shapes use ... to accept both unbatched (1D) and batched (2D) inputs
# when used with apply_tesseract(..., vmap_method="broadcast_all").
class InputSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    z_init: Differentiable[Array[..., Float32]]
    xi_init: Differentiable[Array[..., Float32]]
    z_target: Differentiable[Array[..., Float32]]
    flat_params: Differentiable[Array[..., Float32]] = Field(
        description="Flattened Centralized NN weight vector"
    )

    t_steps: int = Field(default=300)


class OutputSchema(BaseModel):
    z_trajectory: Differentiable[Array[..., Float32]]
    xi_trajectory: Differentiable[Array[..., Float32]]
    u_trajectory: Differentiable[Array[..., Float32]]
    v_trajectory: Differentiable[Array[..., Float32]]


#
# Required endpoints
#


def _solve_single(z_init, xi_init, z_target, flat_params, t_steps):
    """Solve for a single sample."""
    params = _UNFLATTEN_FN(flat_params)
    return solver.solve_with_policy(
        z_init, xi_init, z_target, params, POLICY_MODEL.apply, t_steps
    )


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    z_init = inputs["z_init"]
    xi_init = inputs["xi_init"]
    z_target = inputs["z_target"]
    flat_params = inputs["flat_params"]
    t_steps = inputs["t_steps"]

    ndim = z_init.ndim
    if ndim == 1:
        z_traj, xi_traj, u_traj, v_traj = _solve_single(
            z_init, xi_init, z_target, flat_params, t_steps
        )
    elif ndim == 2:
        solve_fn = lambda z, xi, zt, fp: _solve_single(z, xi, zt, fp, t_steps)
        z_traj, xi_traj, u_traj, v_traj = jax.vmap(solve_fn)(
            z_init, xi_init, z_target, flat_params
        )
    else:
        raise ValueError(f"Expected 1D or 2D z_init, got ndim={ndim}")

    return {
        "z_trajectory": z_traj,
        "xi_trajectory": xi_traj,
        "u_trajectory": u_traj,
        "v_trajectory": v_traj,
    }


def apply(inputs: InputSchema) -> dict:
    return apply_jit(inputs.model_dump())


#
# Jax-handled gradient endpoints (no need to modify)
#


def jacobian(
    inputs: InputSchema,
    jac_inputs: set[str],
    jac_outputs: set[str],
):
    return jac_jit(inputs.model_dump(), tuple(jac_inputs), tuple(jac_outputs))


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    return jvp_jit(
        inputs.model_dump(),
        tuple(jvp_inputs),
        tuple(jvp_outputs),
        tangent_vector,
    )


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    return vjp_jit(
        inputs.model_dump(),
        tuple(vjp_inputs),
        tuple(vjp_outputs),
        cotangent_vector,
    )


def abstract_eval(abstract_inputs):
    """Calculate output shape of apply from the shape of its inputs."""
    is_shapedtype_dict = lambda x: type(x) is dict and (x.keys() == {"shape", "dtype"})
    is_shapedtype_struct = lambda x: isinstance(x, jax.ShapeDtypeStruct)

    jaxified_inputs = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(**x) if is_shapedtype_dict(x) else x,
        abstract_inputs.model_dump(),
        is_leaf=is_shapedtype_dict,
    )
    dynamic_inputs, static_inputs = eqx.partition(
        jaxified_inputs, filter_spec=is_shapedtype_struct
    )

    def wrapped_apply(dynamic_inputs):
        inputs = eqx.combine(static_inputs, dynamic_inputs)
        return apply_jit(inputs)

    jax_shapes = jax.eval_shape(wrapped_apply, dynamic_inputs)
    return jax.tree.map(
        lambda x: (
            {"shape": x.shape, "dtype": str(x.dtype)} if is_shapedtype_struct(x) else x
        ),
        jax_shapes,
        is_leaf=is_shapedtype_struct,
    )


#
# Helper functions
#


@eqx.filter_jit
def jac_jit(
    inputs: dict,
    jac_inputs: tuple[str],
    jac_outputs: tuple[str],
):
    filtered_apply = filter_func(apply_jit, inputs, jac_outputs)
    return jax.jacrev(filtered_apply)(
        flatten_with_paths(inputs, include_paths=jac_inputs)
    )


@eqx.filter_jit
def jvp_jit(
    inputs: dict, jvp_inputs: tuple[str], jvp_outputs: tuple[str], tangent_vector: dict
):
    filtered_apply = filter_func(apply_jit, inputs, jvp_outputs)
    return jax.jvp(
        filtered_apply,
        [flatten_with_paths(inputs, include_paths=jvp_inputs)],
        [tangent_vector],
    )[1]


@eqx.filter_jit
def vjp_jit(
    inputs: dict,
    vjp_inputs: tuple[str],
    vjp_outputs: tuple[str],
    cotangent_vector: dict,
):
    filtered_apply = filter_func(apply_jit, inputs, vjp_outputs)
    _, vjp_func = jax.vjp(
        filtered_apply, flatten_with_paths(inputs, include_paths=vjp_inputs)
    )
    return vjp_func(cotangent_vector)[0]
