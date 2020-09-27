# -*- coding: utf-8 -*-

""" Variational Quantum Eigensolver. """

from qutip.qip.circuit import QubitCircuit


def _estimate_outcome_probabilities(qc, initial_state, runs):
    """ Run a circuit a number of times and estimate the probabilities of
        each outcome.
    """
    outcome_counts = [0, 0]
    for _ in range(runs):
        qc.run(state=initial_state)
        result = qc.cbits[0]
        outcome_counts[result] += 1
    return [outcome_counts[0] / runs, outcome_counts[1] / runs]


def _analytic_outcome_probabilities(qc, initial_state):
    """ Run a circuit a number of times and estimate the probabilities of
        each outcome.
    """
    states, state_probs = qc.run_statistics(initial_state)
    return state_probs


def estimate_energy(
    h_coeffs,
    h_measurement_circuits,
    initial_state,
    ansatz_circuit,
    runs=100,
    analytical=False,
):
    """ Estimate the energy of a given state for a specified Hamiltonian. """
    energy = 0
    N = len(initial_state.dims[0])  # number of qubits
    sign = (N == 1) and 1 or -1
    for indices, coeff in h_coeffs.items():
        qc = QubitCircuit(N=N, num_cbits=1)
        qc.add_circuit(ansatz_circuit)
        qc.add_circuit(h_measurement_circuits[indices])
        qc.add_measurement("M", targets=[0], classical_store=0)
        if indices == "I" * N:
            energy_term = 1
            # TODO: print(indices, coeff, energy_term, "SKIP")
        else:
            if analytical:
                p0, p1 = _analytic_outcome_probabilities(qc, initial_state)
            else:
                p0, p1 = _estimate_outcome_probabilities(qc, initial_state, runs=100)
            energy_term = sign * ((-1 * p0) + (+1 * p1))
            # TODO: print(indices, coeff, energy_term, p0, p1)
        energy += coeff * energy_term
    return energy
