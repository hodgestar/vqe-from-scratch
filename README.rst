vqe-from-scratch
================

Implement VQE from scratch [*]_.

.. [*] For the moment "from scratch" is defined as "using only states, operators and circuits from qutip".

Potential improvements:

* Implement generic calculation of Pauli measurement gates.
* Use a quantum device simulator (e.g. `qutip.qip.device.LinearSpinChain`) to
  emulate the quantum circuit rather than `Circuit.run()`.
