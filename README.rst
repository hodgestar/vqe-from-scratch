vqe-from-scratch
================

Implement VQE from scratch [*]_.

.. [*] For the moment "from scratch" is defined as "using only states, operators and circuits from qutip".

Notebook in nbviewer:

* https://nbviewer.jupyter.org/github/hodgestar/vqe-from-scratch/blob/master/vqefs.ipynb

Potential improvements:

* Implement generic calculation of Pauli measurement gates.
* Use a quantum device simulator (e.g. `qutip.qip.device.LinearSpinChain`) to
  emulate the quantum circuit rather than `Circuit.run()`.

References:

* `A variational eigenvalue solver on a quantum processor <https://arxiv.org/pdf/1304.3061.pdf>`_
* `Microsoft Quantum, Pauli Measurements <https://docs.microsoft.com/en-us/quantum/concepts/pauli-measurements>`_
