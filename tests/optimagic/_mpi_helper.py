"""Driver script for the MPI batch-evaluator integration test.

Run as ``python -m mpi4py.futures _mpi_helper.py`` under ``mpiexec -n N``.
Only the driver rank reaches the code below; ``mpi4py.futures`` parks the
worker ranks inside ``MPIPoolExecutor``, so no rank-guarding is needed here.

Exits 0 when ``mpi_batch_evaluator`` returns the closure's evaluations in
input order, nonzero otherwise.
"""

import sys

from optimagic.batch_evaluators import mpi_batch_evaluator


def main() -> int:
    k = 3
    closure = lambda x: x * k  # noqa: E731

    arguments = [0, 1, 2, 3, 4]
    expected = [x * k for x in arguments]

    calculated = mpi_batch_evaluator(func=closure, arguments=arguments)

    if calculated != expected:
        print(  # noqa: T201
            f"MPI batch evaluator mismatch: got {calculated}, expected {expected}",
            file=sys.stderr,
        )
        return 1

    print(f"MPI batch evaluator ok: {calculated}")  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
