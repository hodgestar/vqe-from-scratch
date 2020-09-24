# -*- coding: utf-8 -*-

""" Tests for vqefs.pauli_decomposition. """

import itertools

import hypothesis.strategies as st
import pytest
from hypothesis import given
from qutip import qeye, sigmax, sigmay, sigmaz, tensor

from vqefs.pauli_decomposition import pauli_decompose, PAULI_INDICES, PAULI_MAP


def h_from_coeffs(coeffs):
    """ Convert a dictionary of coefficients into an operator. """
    H = 0
    for indices, value in coeffs.items():
        H += value * tensor(*[PAULI_MAP[idx] for idx in indices])
    return H


def remove_approximate_zeros(coeffs):
    """ Remove approximate zeros from a dictionary of coefficients. """
    return {k: v for k, v in coeffs.items() if v != pytest.approx(0)}


class TestPauliDecompose:

    # hypothesis strategies

    st_coeff = st.floats(min_value=0.1, max_value=100)

    st_1q_idx = st.sampled_from(PAULI_INDICES)
    st_1q_coeffs = st.lists(
        st.tuples(st_1q_idx, st_coeff),
        min_size=1,
        max_size=4 ** 1,
        unique_by=lambda x: x[0],
    )

    st_2q_idx = st.sampled_from(
        ["".join(idx) for idx in itertools.product(PAULI_INDICES, PAULI_INDICES)]
    )
    st_2q_coeffs = st.lists(
        st.tuples(st_2q_idx, st_coeff),
        min_size=1,
        max_size=4 ** 2,
        unique_by=lambda x: x[0],
    )

    st_3q_idx = st.sampled_from(
        [
            "".join(idx)
            for idx in itertools.product(PAULI_INDICES, PAULI_INDICES, PAULI_INDICES)
        ]
    )
    st_3q_coeffs = st.lists(
        st.tuples(st_3q_idx, st_coeff),
        min_size=1,
        max_size=4 ** 3,
        unique_by=lambda x: x[0],
    )

    def test_single_qubit_operator(self):
        H = qeye(2) + 2 * sigmax() + 3 * sigmay() + 4 * sigmaz()
        coeffs = pauli_decompose(H)
        assert coeffs == {
            "I": 1.0,
            "X": 2.0,
            "Y": 3.0,
            "Z": 4.0,
        }

    @given(st_1q_coeffs)
    def test_many_single_qubit_operators(self, coeff_tuples):
        expected_coeffs = dict(coeff_tuples)
        H = h_from_coeffs(expected_coeffs)
        coeffs = pauli_decompose(H)
        assert remove_approximate_zeros(coeffs) == pytest.approx(expected_coeffs)

    def test_two_qubit_operator(self):
        H = (
            tensor(qeye(2), qeye(2))
            + 2 * tensor(sigmax(), sigmay())
            + 3 * tensor(sigmay(), sigmaz())
            + 4 * tensor(qeye(2), sigmaz())
        )
        coeffs = pauli_decompose(H)
        assert coeffs == {
            "II": 1.0,
            "XY": 2.0,
            "YZ": 3.0,
            "IZ": 4.0,
        }

    @given(st_2q_coeffs)
    def test_many_two_qubit_operators(self, coeff_tuples):
        expected_coeffs = dict(coeff_tuples)
        H = h_from_coeffs(expected_coeffs)
        coeffs = pauli_decompose(H)
        assert remove_approximate_zeros(coeffs) == pytest.approx(expected_coeffs)

    def test_three_qubit_operator(self):
        H = (
            tensor(qeye(2), qeye(2), qeye(2))
            + 2 * tensor(sigmax(), sigmay(), sigmax())
            + 3 * tensor(sigmay(), sigmaz(), sigmaz())
            + 4 * tensor(qeye(2), sigmaz(), qeye(2))
        )
        coeffs = pauli_decompose(H)
        assert coeffs == {
            "III": 1.0,
            "XYX": 2.0,
            "YZZ": 3.0,
            "IZI": 4.0,
        }

    @given(st_3q_coeffs)
    def test_many_three_qubit_operators(self, coeff_tuples):
        expected_coeffs = dict(coeff_tuples)
        H = h_from_coeffs(expected_coeffs)
        coeffs = pauli_decompose(H)
        assert remove_approximate_zeros(coeffs) == pytest.approx(expected_coeffs)
