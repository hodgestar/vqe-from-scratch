---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.6.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Variational Quantum Eigensolver from Scratch

Implement a simple VQE library from scracth using QuTiP Qobjs and circuits and then use it to approximate the lowest eigenstate of a two qubit Hamiltonian (H) defined below.

Library features and limitations:

* Can decompose arbitrary Hamiltonians into Pauli operators (system resources allowing).
* Can only build Pauli measurement operators for one and two qubit operators (needs extending to support arbitrary
  size operators).

```python
import logging

logging.basicConfig(level=logging.INFO)
```

```python
%matplotlib inline

import math
import warnings

import qutip
import scipy.optimize
from qutip.qip.circuit import QubitCircuit

import vqefs.pauli
import vqefs.vqe

# ignore deprecation warning from the lastest ipykernel release -- hopefully the rest of the
# jupyter notebook system will catch up soon.
warnings.filterwarnings("ignore", message=r"`should_run_async`")
```

# Define H

```python
H = qutip.Qobj([
    [1, 0, 0, 0],
    [0, 0, -1, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1],
], dims=[[2, 2], [2, 2]])
H
```

# Decompose H into a sum of Pauli matrix tensor products

* The function vqefs.pauli.decompose(...) returns the coefficients of each term.
* Terms with zero coefficients are excluded.
* The notation "XX" is short for "X ⊗ X".

```python
h_coeffs = vqefs.pauli.decompose(H)
h_coeffs
```

# Construct measurement circuits for each Pauli term

* And display each measurement circuit so that we can inspect them for correctness. :)

```python
h_measurement_circuits = {
    indices: vqefs.pauli.measurement_circuit(indices) for indices in h_coeffs
}
```

```python
h_measurement_circuits["II"]
```

```python
h_measurement_circuits["ZZ"]
```

```python
h_measurement_circuits["XX"]
```

```python
h_measurement_circuits["YY"]
```

# Define the form of the states to optimize over (i.e. the ansatz)

* And display an example of the circuit to check that it looks correct. :)

```python
def h_ansatz(theta):
    qc = QubitCircuit(N=2)

    qc.add_gate("SNOT", targets=0)
    qc.add_gate("CNOT", controls=0, targets=1)
    qc.add_gate("RX", targets=0, arg_value=theta)

    #qc.add_gate("RX", targets=0, arg_value=theta)
    #qc.add_gate("CNOT", controls=0, targets=1)
    #qc.add_gate("SNOT", targets=0)
    
    return qc
    
h_ansatz(math.pi / 2)
```

```python
initial_state = qutip.ket("00")
initial_state
```

# Estimate the minimum energy

```python
# set to analytical=True to use exact outcome probabilitites, set to False to simulate outcomes
analytical = True

def h_energy(x):
    theta, = x
    ansatz_circuit = h_ansatz(theta)
    energy = vqefs.vqe.estimate_energy(
        h_coeffs, h_measurement_circuits,
        initial_state, ansatz_circuit,
        runs=200, analytical=analytical
    )
    return energy

result = scipy.optimize.minimize(
    h_energy, x0=(math.pi / 4),
    bounds=[(-math.pi, math.pi)],
    options={"maxiter": 5, "disp": True},
)
result
```

```python
print(f"minimum energy: {result.fun}")
print(f"best theta: {result.x}")
```

```python
# display final state
state, _prob = h_ansatz(result.x[0]).run(initial_state)
state
```

# Compare with the analytical result

```python
H.eigenstates()
```

# One qubit example for testing

* Test a simple single qubit example to sanity check the algorithm.

```python
h1d_coeffs = {"I": 1, "X": 2, "Y": 3, "Z": 4}
h1d_coeffs
```

```python
h1d = vqefs.pauli.compose(h1d_coeffs)
h1d
```

```python
h1d_measurement_circuits = {
    indices: vqefs.pauli.measurement_circuit(indices) for indices in h1d_coeffs
}
```

```python
def h1d_ansatz(theta1, theta2):
    qc = QubitCircuit(N=1)
    qc.add_gate("RX", targets=0, arg_value=theta1)
    qc.add_gate("RY", targets=0, arg_value=theta2)
    return qc
    
h1d_ansatz(math.pi / 2, math.pi / 2)
```

```python
initial_state = qutip.ket("0")
initial_state
```

```python
# set to analytical=True to use exact outcome probabilitites, set to False to simulate outcomes
analytical = True

def h1d_energy(x):
    theta1, theta2 = x
    ansatz_circuit = h1d_ansatz(theta1, theta2)
    energy = vqefs.vqe.estimate_energy(
        h1d_coeffs, h1d_measurement_circuits,
        initial_state, ansatz_circuit,
        runs=200, analytical=analytical
    )
    return energy

result = scipy.optimize.minimize(
    h1d_energy, x0=(math.pi / 4, math.pi / 4),
    bounds=[(-math.pi, math.pi), (-math.pi, math.pi)],
    options={"maxiter": 10, "disp": True},
)
result
```

```python
print(f"minimum energy: {result.fun}")
print(f"best theta: {result.x}")
```

```python
# display final state
h1d_ansatz(*result.x).run(initial_state)[0]
```

```python
h1d.eigenstates()
```

# Two qubit example for testing

* Test a simple two qubit example to sanity check the algorithm.

```python
h2d_coeffs = {"II": 1, "XX": 2, "YY": 3, "ZZ": 4}
h2d_coeffs
```

```python
h2d = vqefs.pauli.compose(h2d_coeffs)
h2d
```

```python
h2d_measurement_circuits = {
    indices: vqefs.pauli.measurement_circuit(indices) for indices in h2d_coeffs
}
```

```python
def h2d_ansatz(theta1, theta2, theta3, theta4):
    """ A fairly arbitrary 2-qubit ansatz. """
    qc = QubitCircuit(N=2)
    qc.add_gate("RY", targets=0, arg_value=theta1)    
    qc.add_gate("CNOT", controls=1, targets=0)
    qc.add_gate("RY", targets=1, arg_value=theta2)
    qc.add_gate("RZ", targets=0, arg_value=theta3)
    qc.add_gate("CNOT", controls=0, targets=1)
    qc.add_gate("RY", targets=1, arg_value=theta4)
    return qc
    
h2d_ansatz(math.pi / 2, math.pi / 2, math.pi / 2, math.pi / 2)
```

```python
initial_state = qutip.ket("00")
initial_state
```

```python
# set to analytical=True to use exact outcome probabilitites, set to False to simulate outcomes
analytical = True

def h2d_energy(x):
    theta1, theta2, theta3, theta4 = x
    ansatz_circuit = h2d_ansatz(theta1, theta2, theta3, theta4)
    energy = vqefs.vqe.estimate_energy(
        h2d_coeffs, h2d_measurement_circuits,
        initial_state, ansatz_circuit,
        runs=200, analytical=analytical
    )
    return energy

result = scipy.optimize.minimize(
    h2d_energy, x0=(math.pi / 4, math.pi / 4, math.pi / 4, math.pi / 4),
    bounds=[(-math.pi, math.pi)] * 4,
    options={"maxiter": 20, "disp": True},
)
result
```

```python
print(f"minimum energy: {result.fun}")
print(f"best theta: {result.x}")
```

```python
# display final state
h2d_ansatz(*result.x).run(initial_state)[0]
```

```python
h2d.eigenstates()
```

```python

```