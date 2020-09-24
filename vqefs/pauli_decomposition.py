# -*- coding: utf-8 -*-

""" Decompose a Hamiltonian into tensor products of Pauli matrices. """


def pauli_decompose(H):
    """ Decompose the Hamiltonian H into tensor products of Pauli matrices.

        A Hamiltonian H may be decomposed into a sum of products of Pauli
        matrices as follows:

            H = ∑ a_{ij..k} σ_i * σ_j * ... * σ_k

        where the coefficients are given by:

            a_{ij..k} = ¼ tr (σ_i * σ_j * ... * σ_k) H)

        and the indices i, j, .., k run through the four Pauli matrices I, Z,
        X and Y:

            i, j, .., k ∈ {I, Z, X, Y}

        :parameter ??? H:
            The Hamiltonian to decompose.

        :returns:
            The coefficients of the decomposition.
    """
    raise NotImplementedError()
