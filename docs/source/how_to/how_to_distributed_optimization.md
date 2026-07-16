(distributed-optimization)=

# How to run an optimization across multiple machines with MPI

On a single machine, optimagic parallelizes the batched criterion evaluations that an
algorithm requests (for example, the sampled points of a trust-region optimizer such as
tranquilo) across local processes or threads. On an HPC cluster you usually want those
evaluations spread across many nodes instead. The right tool for that is MPI, and
optimagic ships a batch evaluator built on `mpi4py.futures.MPIPoolExecutor`.

## The single-driver model

The optimizer itself is inherently sequential: it proposes a batch of parameter vectors,
waits for their criterion values, and then proposes the next batch. Running several
copies of the optimizer on different ranks would not help — they would all do the same
work. The correct pattern is:

- **One driver rank** runs the optimizer and your `minimize` / `maximize` call.
- **All other ranks are workers**, parked by the `mpi4py.futures` launcher, waiting to
  evaluate the criterion on the parameter vectors the driver sends them.

This is exactly what the `python -m mpi4py.futures` launcher sets up.

## Installation

The MPI batch evaluator needs the optional `mpi4py` dependency:

```bash
pip install "optimagic[mpi]"
```

You also need a working MPI implementation (for example, Open MPI or MPICH) on the
cluster — typically provided as a module.

## Launch precondition

The MPI batch evaluator only works if the program was started with the `mpi4py.futures`
launcher so that worker ranks exist. On a SLURM cluster this looks like:

```bash
srun python -m mpi4py.futures your_script.py
```

or, outside of SLURM:

```bash
mpiexec -n <N> python -m mpi4py.futures your_script.py
```

If you forget the launcher, the MPI batch evaluator raises a clear error pointing you at
this requirement.

## Using the MPI batch evaluator

Inside `your_script.py`, just pass `batch_evaluator="mpi"`:

```python
import optimagic as om


def sphere(params):
    return params @ params


res = om.minimize(
    fun=sphere,
    params=[1.0, 2.0, 3.0],
    algorithm="tranquilo",
    batch_evaluator="mpi",
)
```

The same `MPIPoolExecutor` is created on the first batch and reused for every subsequent
batch, so the worker ranks stay alive for the whole optimization.

## Using any executor directly

`batch_evaluator="mpi"` is a convenience wrapper. If you want full control over the
executor — its worker count, an existing pool, or a non-MPI executor — build a batch
evaluator from any `concurrent.futures.Executor` with `executor_batch_evaluator`:

```python
from mpi4py.futures import MPIPoolExecutor

from optimagic.batch_evaluators import executor_batch_evaluator

res = om.minimize(
    fun=sphere,
    params=[1.0, 2.0, 3.0],
    algorithm="tranquilo",
    batch_evaluator=executor_batch_evaluator(MPIPoolExecutor()),
)
```

`executor_batch_evaluator` also works with a `ProcessPoolExecutor` or
`ThreadPoolExecutor`, which is handy for testing the distributed code path on a single
machine. Closures and locally defined criterion functions are transported correctly
because the evaluator serializes them with `cloudpickle`.
