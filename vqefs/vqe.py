# -*- coding: utf-8 -*-

""" Variational Quantum Eigensolver. """

import logging

from qutip.qip.circuit import QubitCircuit


logger = logging.getLogger(__name__)


def _estimate_outcome_probabilities(qc, initial_state, shots):
    """ Run a circuit a number of times and estimate the probabilities of
        each outcome.

        :param QubitCircuit qc:
            The circuit to run. The circuit should contain a single
            measurement and store its result in `.cbits[0]`.
        :param Qobj initial_state:
            The initial state to run the circuit on.
        :param int shots:
            The number of runs to perform.

        :return list:
            A list containing the fraction of runs that returned measurements
            0 and 1.
    """
    outcome_counts = [0, 0]
    for _ in range(shots):
        qc.run(state=initial_state)
        result = qc.cbits[0]
        outcome_counts[result] += 1
    return [outcome_counts[0] / shots, outcome_counts[1] / shots]


def _analytic_outcome_probabilities(qc, initial_state):
    """ Calculate the probabilities of each outcome analytically.

        :param QubitCircuit qc:
            The circuit to run. The circuit should contain a single
            measurement and store its result in `.cbits[0]`.
        :param Qobj initial_state:
            The initial state to run the circuit on.

        :return list:
            A list containing the probabilities of measurement outcomes 0 and 1.
    """
    states, state_probs = qc.run_statistics(initial_state)
    return state_probs


def estimate_energy(
    h_coeffs,
    h_measurement_circuits,
    initial_state,
    ansatz_circuit,
    shots=100,
    analytical=False,
):
    """ Estimate the energy of a given state for a specified Hamiltonian.

        :param dict h_coeffs:
            The Pauli decomposition of the Hamiltonian. This specifies the
            Hamiltonian. If you have an operator `H` then
            `vqefs.pauli.decompse(H)` will return the required dictionary of
            coefficients.
        :param dict h_measurement_circuits:
            The dictionary of measurement operators for each of the Pauli
            coefficients in `h_coeffs`. If you have a dictionary of coefficients,
            the measurement circuits may be constructed using::

                h_measurement_circuits = {
                    indices: vqefs.pauli.measurement_circuit(indices)
                    for indices in h2d_coeffs
                }
        :param Qobj initial_state:
            The initial state to pass to the completed circuit when it is run.
        :param QubitCircuit ansatz_circuit:
            The circuit used to evolve the initial_state into the prepared state
            which will be measured.
        :param int shots:
            How many times to run and measure each circuit. Default: 100.
        :param bool analytical:
            If True, calculate the probabilities analytically using
            `QubitCircuit.run_statistics(...)`. If False, run the
            circuit multiple times with `QubitCircuit.run(...)`.
            Default: False.

        :return float:
            The estimated energy of the prepared state.
    """
    energy = 0
    N = len(initial_state.dims[0])  # number of qubits
    sign = (N == 1) and 1 or -1
    for indices, coeff in h_coeffs.items():
        if indices == "I" * N:
            energy_term = 1
            logger.info(
                "%s: coeff: %g, energy: %g", indices, coeff, energy_term,
            )
        else:
            qc = QubitCircuit(N=N, num_cbits=1)
            qc.add_circuit(ansatz_circuit)
            qc.add_circuit(h_measurement_circuits[indices])
            qc.add_measurement("M", targets=[0], classical_store=0)
            if analytical:
                p0, p1 = _analytic_outcome_probabilities(qc, initial_state)
            else:
                p0, p1 = _estimate_outcome_probabilities(qc, initial_state, shots=shots)
            energy_term = sign * ((-1 * p0) + (+1 * p1))
            logger.info(
                "%s: coeff: %g, energy: %g, probabilities: (%g, %g)",
                indices,
                coeff,
                energy_term,
                p0,
                p1,
            )
        energy += coeff * energy_term
    return energy
