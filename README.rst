vqe-from-scratch
================

An implementation of VQE from scratch [*]_.

.. [*] For the moment "from scratch" is defined as "using only states, operators and circuits from qutip".

Notebook with solution to Screening Task 4
------------------------------------------

* https://nbviewer.jupyter.org/github/hodgestar/vqe-from-scratch/blob/master/vqefs.ipynb

**Click the link above to view the notebook with the solution!**

Library features
----------------

* Can decompose Hamiltonians into Pauli operators.
* Can re-compose Pauli decompositions into Hamiltonians.
* Can build Pauli measurement operators for one and two qubit operators (needs extending to support arbitrary
  size operators).
* Can estimate the energy of a state for a given (decomposed) Hamiltonian either by:

  * simulation (mutiple calls to `QubitCircuit.run`)
  * analytically (a single call to `QubitCircuit.run_statistics`).

* Extensive unit tests for `vqefs.pauli.decompose` using `hypothesis <https://hypothesis.readthedocs.io/>`_.

Future work
-----------

* It would be nice to run the circuit on a QuTiP Processor instead (i.e. a simulator for a real quantum device)
  but this requires adding support for circuits with measurement to the Processors (likely not too hard, but
  not done yet).
* Implement calculation of Pauli measurement gates for 3+ qubits.
* Extend the extensive unit tests to the rest of the library.

Note
----

* This library requires the QuTiP master branch for QubitCircuit measurement support. Once QuTiP 4.6 is released
  it should be possilbe to run in on QuTiP 4.6.
* Full disclosure -- I implemented some of the initial parts of QuTiP's measurement support.

References
----------

* `A variational eigenvalue solver on a quantum processor <https://arxiv.org/pdf/1304.3061.pdf>`_
* `Microsoft Quantum, Pauli Measurements <https://docs.microsoft.com/en-us/quantum/concepts/pauli-measurements>`_
