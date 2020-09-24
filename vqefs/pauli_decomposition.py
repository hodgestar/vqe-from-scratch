# -*- coding: utf-8 -*-

""" Decompose a Hamiltonian into tensor products of Pauli matrices. """

import itertools

from qutip import qeye, sigmax, sigmay, sigmaz, tensor


PAULI_MAP = {
    "I": qeye(2),
    "Z": sigmaz(),
    "X": sigmax(),
    "Y": sigmay(),
}

PAULI_INDICES = list(PAULI_MAP.keys())


def pauli_decompose(H):
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

        :returns dict:
            A dictionary of the coefficients of the decomposition. The keys
            are strings representing the indices and the values are the
            coefficients corresponding to those indices. E.g. `"XY": 5.0`
            says that `H = ... + 5.0 σ_X * σ_Y + ...`.

            Coefficients that are zero are omitted from the dictionary.
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
        if a != 0.0:
            coeffs["".join(indices)] = a
    return coeffs
