# -*- coding: utf-8 -*-

""" Utilities for working with Pauli decompositions. """

import itertools
import math

from qutip import Qobj, qeye, sigmax, sigmay, sigmaz, tensor
from qutip.qip.circuit import QubitCircuit


PAULI_MAP = {
    "I": qeye(2),
    "Z": sigmaz(),
    "X": sigmax(),
    "Y": sigmay(),
}

PAULI_INDICES = list(PAULI_MAP.keys())


def decompose(H, tol=1e-12):
    """ Decompose the Hamiltonian H into tensor products of Pauli matrices.

        A Hamiltonian H may be decomposed into a sum of products of Pauli
        matrices as follows:

            H = ∑ a_{ij..k} σ_i * σ_j * ... * σ_k

        where the coefficients are given by:

            a_{ij..k} = (1/2^n) tr (σ_i * σ_j * ... * σ_k) H)

        and the indices i, j, .., k run through the four Pauli matrices I, Z,
        X and Y:

            i, j, .., k ∈ {I, Z, X, Y}

        and n is the number of indices.

        :parameter qobj H:
            The Hermitian operator to decompose.
        :parameter float tol:
            The threshold for determining whether a coefficient is zero and
            should be omitted from the result. Default: 1e-12.

        :returns dict:
            A dictionary of the coefficients of the decomposition. The keys
            are strings representing the indices and the values are the
            coefficients corresponding to those indices. E.g. `"XY": 5.0`
            says that `H = ... + 5.0 σ_X * σ_Y + ...`.

            Coefficients that are approximagely zero are omitted.
    """
    assert H.isherm, "Pauli decomposition requires H to be a Hermitian operator"
    dims = H.dims[0]
    assert all(
        x == 2 for x in dims
    ), "Pauli decompsition requires H to be an operator on qubits"
    n = len(dims)  # number of qubits
    inv_d = 1.0 / (2 ** n)
    coeffs = {}
    for indices in itertools.product(*([PAULI_INDICES] * n)):
        op = tensor(*[PAULI_MAP[idx] for idx in indices])
        a = inv_d * (op * H).tr()
        if abs(a) >= tol:
            coeffs["".join(indices)] = a
    return coeffs


def compose(coeffs):
    """ Sum Pauli decomposition into an operator.

        :parameter dict coeffs:
            The Pauli decompostion coefficients dictionary as returned by
            `vqefs.pauli.decompose(...)`.

        :returns qobj:
            The Hermitian operator that is the sum of the Pauli decomposition
            terms.
    """
    H = 0
    for indices, coeff in coeffs.items():
        H += coeff * tensor(*[PAULI_MAP[idx] for idx in indices])
    return H


PAULI_MEASUREMENT_CIRCUITS_1Q = {
    # There is no measurement circuit for "I" because it has only a single
    # eigenstate (+1).
    "Z": [],
    "X": [("SNOT", {"targets": 0})],
    "Y": [
        ("PHASEGATE", {"targets": 0, "arg_value": -math.pi / 2}),
        ("SNOT", {"targets": 0}),
    ],
}
assert len(PAULI_MEASUREMENT_CIRCUITS_1Q) == 3

PAULI_MEASUREMENT_CIRCUITS_2Q = {
    # There is no measurement circuit for "II" because it has only a single
    # eigenstate (+1).
    "IZ": [("SWAP", {"targets": [0, 1]})],
    "IX": [("SWAP", {"targets": [0, 1]}), ("SNOT", {"targets": 0})],
    "IY": [
        ("SWAP", {"targets": [0, 1]}),
        ("PHASEGATE", {"targets": 0, "arg_value": -math.pi / 2}),
        ("SNOT", {"targets": 0}),
    ],
    "ZI": [],
    "ZZ": [("CNOT", {"controls": 1, "targets": 0})],
    "ZX": [("SNOT", {"targets": 1}), ("CNOT", {"controls": 1, "targets": 0})],
    "ZY": [
        ("PHASEGATE", {"targets": 1, "arg_value": -math.pi / 2}),
        ("SNOT", {"targets": 1}),
        ("CNOT", {"controls": 1, "targets": 0}),
    ],
    "XI": [("SNOT", {"targets": 0})],
    "XX": [
        ("SNOT", {"targets": 0}),
        ("SNOT", {"targets": 1}),
        ("CNOT", {"controls": 1, "targets": 0}),
    ],
    "XZ": [("SNOT", {"targets": 0}), ("CNOT", {"controls": 1, "targets": 0})],
    "XY": [
        ("PHASEGATE", {"targets": 1, "arg_value": -math.pi / 2}),
        ("SNOT", {"targets": 1}),
        ("SNOT", {"targets": 0}),
        ("CNOT", {"controls": 1, "targets": 0}),
    ],
    "YI": [
        ("PHASEGATE", {"targets": 0, "arg_value": -math.pi / 2}),
        ("SNOT", {"targets": 0}),
    ],
    "YY": [
        ("PHASEGATE", {"targets": 0, "arg_value": -math.pi / 2}),
        ("SNOT", {"targets": 0}),
        ("PHASEGATE", {"targets": 1, "arg_value": -math.pi / 2}),
        ("SNOT", {"targets": 1}),
        ("CNOT", {"controls": 1, "targets": 0}),
    ],
    "YZ": [
        ("PHASEGATE", {"targets": 0, "arg_value": -math.pi / 2}),
        ("SNOT", {"targets": 0}),
        ("CNOT", {"controls": 1, "targets": 0}),
    ],
    "YX": [
        ("SNOT", {"targets": 1}),
        ("PHASEGATE", {"targets": 0, "arg_value": -math.pi / 2}),
        ("SNOT", {"targets": 0}),
        ("CNOT", {"controls": 1, "targets": 0}),
    ],
}
assert len(PAULI_MEASUREMENT_CIRCUITS_2Q) == 15

PAULI_MEASUREMENT_CIRCUITS = {
    1: PAULI_MEASUREMENT_CIRCUITS_1Q,
    2: PAULI_MEASUREMENT_CIRCUITS_2Q,
}


def measurement_circuit(indices):
    """ Return a measurement circuit for the given Pauli decomposition term.

        :param str indices:
            The Pauli terms to return a measurement circuit for. The
            returned circuit rotates the +1 and -1 eigenspaces of the
            Pauli term into the computational basis.

        :return QubitCircuit:
            A circuit that will rotate the Pauli measurement into the
            computational basis.

        Note: If the indices are "I" or "II" or "III..." this function will
        return `None`. These terms have only a (repeated) +1 eigenvalue and
        so cannot be measured in the same way as the other terms. Passing the
        identity terms is supported here for convenience, but measurement
        on these terms always returned the eigenvalue +1.
    """
    n = len(indices)
    assert n in PAULI_MEASUREMENT_CIRCUITS
    if indices == "I" * n:
        return None
    qc = QubitCircuit(N=n)
    for gate, kwargs in PAULI_MEASUREMENT_CIRCUITS[n][indices]:
        qc.add_gate(gate, **kwargs)
    return qc


def xxx_pauli_measurement_gate(indices):
    """ Return a Pauli measurement gate for the Pauli decomposition term
        specified by the indices.

        :param str indices:
            A string representing the decomposition term. E.g.
            "XY" represents σ_X * σ_Y, "ZZI" represents σ_Z * σ_Z * I, etc.

        :return Qobj:
            An operator that transforms the eigenstates of the given term
            into the computational basis.
    """
    op = tensor(*[PAULI_MAP[idx] for idx in indices])
    eigenvalues, eigenstates = op.eigenstates()
    eigen_op = Qobj([ev.data.flatten() for ev in eigenstates])
    return eigen_op.inv()


def xxx_pauli_measurement_gates(n):
    """ Return a complete set of Pauli measurement gates for the given
        number of qubits.

        Each gate rotates the eigenstates of the given Pauli decomposition
        term into the computational basis.
    """
    gates = {}
    for indices in itertools.product(*([PAULI_INDICES] * n)):
        indices = "".join(indices)
        gates[f"PM_{indices}"] = lambda: xxx_pauli_measurement_gate(indices)
    return gates
