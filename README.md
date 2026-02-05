# Tesseract Core Performance Reproduction (Heat 1D Decentralized)

This branch, `repro-performance-issue`, is dedicated to isolating a performance bottleneck encountered when training PDE solvers using Tesseract Core. We suspect the primary driver of latency is the serialization overhead inherent in the Tesseract communication protocol.

For clarity and reproducibility, we have pruned the repository to retain only a minimal test case: the **Heat 1D decentralized** example. Note that we encountered this same performance degradation across all other PDE solvers tested.

## Getting Started

Follow these steps to set up the environment, build the component, and reproduce the performance behavior.

### 1. Environment Setup

Clone the repository (targeting the reproduction branch) and initialize a virtual environment:

```bash
git clone -b repro-performance-issue [https://github.com/SOLARIS-JHU/Multi-Agent-DPC](https://github.com/SOLARIS-JHU/Multi-Agent-DPC)
cd Multi-Agent-DPC
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Build the Tesseract Component
Navigate to the Tesseract definition directory and build the Docker container image:
```
cd tesseracts/solverHeat_decentralized
tesseract build .
cd ../..
```

Verify that the Tesseract image was built successfully:
```
docker images | grep solver
```

### 3. Run the Training Reproduction
Execute the training script to observe the performance behavior:

```
cd examples/heat1d/decentralized
python train.py
```

### 4. Repository Structure & Navigation
To facilitate navigation, the repository is structured as follows:
- tesseracts/: Contains the Heat Equation solver logic and the corresponding tesseract_api definition.
- examples/heat1d/decentralized/: Contains the minimal reproduction scripts:
- dynamics_dual.py: Exposes the `PDEDynamics` class. This includes a use_tesseract argument to toggle between the Tesseract-wrapped solver and the native JAX solver.
- train.py: The main training script. It instantiates the `PDEDynamics` class (around line 87) to invoke the solver.

Observation: When `use_tesseract=True` is enabled, the training loop exhibits the significant latency we attribute to JSON serialization overhead.

## 1. Performance Hypothesis: The Serialization Bottleneck
We hypothesize that the significant training latency is caused by the serialization and deserialization of massive simulation data at every time step of the training loop.

Our analysis of the Tesseract Core architecture suggests the following contributing factors (NOTE that we're not pro SWE so our understanding might be limited):

- Mandatory Network Protocol: Tesseract components function as "served components" that communicate via HTTP/REST APIs (e.g., POST /apply). Even when the orchestrator and solver are co-located on the same machine, data must be serialized to cross the network/process boundary, rather than utilizing shared memory.

- Validation Overhead: Tesseract Core strictly defines inputs and outputs using Pydantic models to ensure external data validation. For the massive tensors required in PDE simulations (e.g., dense grid states), validating and converting these objects at every step introduces substantial computational overhead.

- Distributed-First Architecture: The system is designed for distributed contexts and remote procedure calls (RPC). While this facilitates cloud deployment, it imposes a "network-first" data path that is suboptimal for tight, iterative training loops where low latency is critical.

In this Heat 1D example, the solver state must be serialized, transmitted to the Tesseract container, processed, and the result (next state or gradient) serialized and returned. This repeated "ping-pong" of large payloads effectively starves the compute resources, making serialization the dominant factor in total runtime.

## 2. Suggested Mitigation Strategies
We stress out that we're not pro SWE, hence take the following just as potential directions. We were thinking that to address this bottleneck while maintaining the benefits of the Tesseract ecosystem, we considered the following optimizations:

### A. Binary Serialization Protocols
Strategy: Investigate extending Tesseract Core to support efficient binary serialization formats instead of standard JSON. 

Rationale: Some binary formats are faster to parse and produce smaller payloads for numerical data, reducing the I/O burden compared to text-based JSON.

### B. Direct Module Import (Local Bypass)
Strategy: For local development and debugging, it would be nice to have a way to bypass the Tesseract Runtime's HTTP interface.

Rationale: Directly importing the underlying tesseract_api.py module in the training script allows for direct memory access. While this bypasses the container isolation, it eliminates the network stack overhead during the experimentation phase.