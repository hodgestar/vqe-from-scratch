# -*- coding: utf-8 -*-

""" Tests for vqefs.pauli. """

import itertools

import hypothesis.strategies as st
import pytest
from hypothesis import given
from qutip import qeye, sigmax, sigmay, sigmaz, tensor

from vqefs.pauli import decompose, PAULI_INDICES, PAULI_MAP


class NQubitOperator:
    """ Test helper to hold an example operator constructed from a decomposition.
        for hypothesis tests.

        :param Qobj H:
            The constructed operator.
        :param dict coeffs:
            The Pauli coefficients used to generate the example operator.
    """

    def __init__(self, H, coeffs):
        self.H = H
        self.coeffs = coeffs


class NQubitOperatorStrategy:
    """ A strategy for generating N-qubit example operators.

        :param int n:
            The number of qubits the example operator should act on.
    """

    def __init__(self, n):
        self.n = n

    def _coeffs_to_h(self, coeffs):
        """ Convert a dictionary of coefficients into an operator. """
        H = 0
        for indices, value in coeffs.items():
            H += value * tensor(*[PAULI_MAP[idx] for idx in indices])
        return H

    def _coeffs_to_op(self, coeffs):
        """ Convert decomposition coefficients to an NQubitOperator instance. """
        return NQubitOperator(H=self._coeffs_to_h(coeffs), coeffs=coeffs)

    def __call__(self):
        """ Return a hypothesis strategy for an example n-qubit operator. """
        st_idx = st.sampled_from(
            ["".join(idx) for idx in itertools.product(*[PAULI_INDICES] * self.n)]
        )
        st_coeff = st.floats(min_value=0.1, max_value=100)
        st_coeffs = st.lists(
            st.tuples(st_idx, st_coeff),
            min_size=1,
            max_size=4 ** self.n,
            unique_by=lambda x: x[0],
        ).map(dict)
        return st_coeffs.map(self._coeffs_to_op)


class TestDecompose:
    def test_1q_op_simple(self):
        """ Test a smple 1-qubit operator. """
        H = qeye(2) + 2 * sigmax() + 3 * sigmay() + 4 * sigmaz()
        coeffs = decompose(H)
        assert coeffs == {
            "I": 1.0,
            "X": 2.0,
            "Y": 3.0,
            "Z": 4.0,
        }

    @given(NQubitOperatorStrategy(n=1)())
    def test_1q_op_many(self, op):
        """ Test a variety of 1-qubit operators. """
        coeffs = decompose(op.H)
        assert coeffs == pytest.approx(op.coeffs)

    def test_2q_op_simple(self):
        """ Test a smple 2-qubit operator. """
        H = (
            tensor(qeye(2), qeye(2))
            + 2 * tensor(sigmax(), sigmay())
            + 3 * tensor(sigmay(), sigmaz())
            + 4 * tensor(qeye(2), sigmaz())
        )
        coeffs = decompose(H)
        assert coeffs == {
            "II": 1.0,
            "XY": 2.0,
            "YZ": 3.0,
            "IZ": 4.0,
        }

    @given(NQubitOperatorStrategy(n=2)())
    def test_2q_op_many(self, op):
        """ Test a variety of 2-qubit operators. """
        coeffs = decompose(op.H)
        assert coeffs == pytest.approx(op.coeffs)

    def test_3q_op_simple(self):
        """ Test a smple 3-qubit operator. """
        H = (
            tensor(qeye(2), qeye(2), qeye(2))
            + 2 * tensor(sigmax(), sigmay(), sigmax())
            + 3 * tensor(sigmay(), sigmaz(), sigmaz())
            + 4 * tensor(qeye(2), sigmaz(), qeye(2))
        )
        coeffs = decompose(H)
        assert coeffs == {
            "III": 1.0,
            "XYX": 2.0,
            "YZZ": 3.0,
            "IZI": 4.0,
        }

    @given(NQubitOperatorStrategy(n=3)())
    def test_3q_op_many(self, op):
        """ Test a variety of 3-qubit operators. """
        coeffs = decompose(op.H)
        assert coeffs == pytest.approx(op.coeffs)
