# -*- coding: utf-8 -*-

""" Tests for vqefs.pauli_decomposition. """

from qutip import qeye, sigmax, sigmay, sigmaz, tensor

from vqefs.pauli_decomposition import pauli_decompose


class TestPauliDecompose:
    def test_single_qubit(self):
        H = qeye(2) + 2 * sigmax() + 3 * sigmay() + 4 * sigmaz()
        coeffs = pauli_decompose(H)
        assert coeffs == {
            "I": 1.0,
            "X": 2.0,
            "Y": 3.0,
            "Z": 4.0,
        }

    def test_two_qubits(self):
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

    def test_three_qubits(self):
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
