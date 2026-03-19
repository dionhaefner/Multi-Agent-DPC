"""No-op Tesseract with identical schema to solverHeat_decentralized.

Returns zeros for all outputs — used to measure pure framework overhead
(data copying, Pydantic validation, callback machinery) without compute.
"""
import numpy as np
import jax.numpy as jnp
from pydantic import BaseModel, Field, ConfigDict
from tesseract_core.runtime import Array, Differentiable, Float32, ShapeDType


class InputSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    z_init: Differentiable[Array[..., Float32]]
    xi_init: Differentiable[Array[..., Float32]]
    z_target: Differentiable[Array[..., Float32]]
    flat_params: Differentiable[Array[..., Float32]]
    t_steps: int = Field(default=300)


class OutputSchema(BaseModel):
    z_trajectory: Differentiable[Array[..., Float32]]
    xi_trajectory: Differentiable[Array[..., Float32]]
    u_trajectory: Differentiable[Array[..., Float32]]
    v_trajectory: Differentiable[Array[..., Float32]]


def _output_shapes(z_init, xi_init, t_steps):
    """Derive output shapes from inputs."""
    if z_init.ndim == 1:
        z_dim, n_agents = z_init.shape[0], xi_init.shape[0]
        return (t_steps, z_dim), (t_steps, n_agents)
    else:
        batch, z_dim = z_init.shape[0], z_init.shape[1]
        n_agents = xi_init.shape[1]
        return (batch, t_steps, z_dim), (batch, t_steps, n_agents)


def apply(inputs: InputSchema) -> OutputSchema:
    z_shape, a_shape = _output_shapes(inputs.z_init, inputs.xi_init, inputs.t_steps)
    return OutputSchema(
        z_trajectory=np.zeros(z_shape, dtype=np.float32),
        xi_trajectory=np.zeros(a_shape, dtype=np.float32),
        u_trajectory=np.zeros(a_shape, dtype=np.float32),
        v_trajectory=np.zeros(a_shape, dtype=np.float32),
    )


def abstract_eval(abstract_inputs):
    t_steps = abstract_inputs.t_steps
    shape = abstract_inputs.z_init.shape
    if len(shape) == 1:
        z_dim, n_agents = shape[0], abstract_inputs.xi_init.shape[0]
        prefix = ()
    else:
        z_dim, n_agents = shape[1], abstract_inputs.xi_init.shape[1]
        prefix = (shape[0],)
    return {
        "z_trajectory": ShapeDType(shape=(*prefix, t_steps, z_dim), dtype="float32"),
        "xi_trajectory": ShapeDType(shape=(*prefix, t_steps, n_agents), dtype="float32"),
        "u_trajectory": ShapeDType(shape=(*prefix, t_steps, n_agents), dtype="float32"),
        "v_trajectory": ShapeDType(shape=(*prefix, t_steps, n_agents), dtype="float32"),
    }


_INPUT_NAMES = ("z_init", "xi_init", "z_target", "flat_params")

def vector_jacobian_product(inputs, vjp_inputs, vjp_outputs, cotangent_vector):
    # Return zeros with correct shapes — no actual compute
    shapes = {
        "z_init": inputs.z_init.shape,
        "xi_init": inputs.xi_init.shape,
        "z_target": inputs.z_target.shape,
        "flat_params": inputs.flat_params.shape,
    }
    return {k: jnp.zeros(shapes[k]) for k in vjp_inputs if k in shapes}
