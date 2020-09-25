# -*- coding: utf-8 -*-

""" Decompose a Hamiltonian into tensor products of Pauli matrices. """

import itertools

from qutip import Qobj, qeye, sigmax, sigmay, sigmaz, tensor


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


def pauli_measurement_gate(indices):
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


def pauli_measurement_gates(n):
    """ Return a complete set of Pauli measurement gates for the given
        number of qubits.

        Each gate rotates the eigenstates of the given Pauli decomposition
        term into the computational basis.
    """
    gates = {}
    for indices in itertools.product(*([PAULI_INDICES] * n)):
        indices = "".join(indices)
        gates[f"PM_{indices}"] = pauli_measurement_gate(indices)
