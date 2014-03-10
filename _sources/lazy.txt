Lazy Evaluation
===============

Approach
--------

Unless explicitly set up to execute greedily, :func:`parallel loops
<pyop2.par_loop>` are not executed immediatly. In doing so, it opens PyOP2 to
a range of execution rescheduling and program transformations at runtime.

Delaying computations
---------------------

In order to delay the execution of parallel loops,
:class:`~pyop2.base.ParLoop` objects inherit from
:class:`~pyop2.base.LazyComputation` and must implement the execution of the
parloop in the :meth:`_run` method.  :class:`~pyop2.base.LazyComputation`
objects implement :meth:`~pyop2.base.LazyComputation.enqueue`, which stores
the computation to be delayed into a list structure,
:class:`~pyop2.base.ExecutionTrace`, for later execution.

Declaring computations
----------------------

For the purpose of reordering the execution of parallel loops while preserving
the correctness of the PyOP2 program, when inheriting from
:class:`~pyop2.base.LazyComputation`, :class:`~pyop2.base.ParLoop` objects
must declare their input/output dependencies, that is the set of
:class:`~pyop2.Dat` objects that read and/or written during the execution of
the :func:`~pyop2.par_loop`. This is directly derived from the arguments of
the :class:`~pyop2.base.ParLoop`, see the instance attributes :attr:`reads`
and :attr:`writes` of :class:`~pyop2.base.LazyComputation`.

Forcing computations
--------------------

To access the up-to-date content of :class:`Dats <pyop2.Dat>`, these must
force the execution of delayed parallel loops before it can be accessed. This
is done by calling :meth:`~pyop2.base.ExecutionTrace.evaluate` of the
:class:`~pyop2.base.ExecutionTrace` from the :class:`~pyop2.Dat`'s public
accessors, passing the reads and writes dependencies to be updated. For
instance ``evaluate(reads={a,b}, writes={b}``, tells ``a`` and ``b`` should be
updated with the intent of being read, and read and writen respectively.

Propagating dependencies
------------------------

The method :meth:`~pyop2.base.ExecutionTrace.evaluate` determines which of the
delayed computations must now be executed in order to satisfy the read and
write dependencies of the given arguments. This method iterates the delayed
execution trace in reverse order: from the most to the least recently delayed
computation.

Let us call :math:`R_a`, :math:`W_a`, :math:`R_c` and :math:`W_c`, the read
(:math:`R`) and write (:math:`W`) dependencies passed as arguments (:math:`a`)
and of the delayed computation (:math:`c`) respectively. Computation
:math:`c` is required to be executed if

.. math::

    R_a \cap W_c \cup W_a \cap R_c \cup W_a \cap W_c

is not empty. In other words, if a dependency read is writen by :math:`c`, if
a dependency written must be read first, or if a dependency is being
overwritten, preserving write ordering.

if `c` is required for :math:`R_a` or :math:`W_a` then :math:`R_a` and
:math:`W_a` become:

.. math::

    R_a = R_a \cup R_c \setminus W_c \\
    W_a = W_a \cup W_c

New write dependencies need to be propagated, but read dependencies that will
be updated need not.

Once the iteration of the list is over, dependent computations are executed in
oldest to most recent order and removed from the list.

Notes
-----

lazy-split branch
~~~~~~~~~~~~~~~~~

Changes:

* A :class:`~pyop2.base.ParLoop` no longer inherits from
  :class:`~pyop2.base.LazyComputation`, instead, the
  :class:`~pyop2.base.ParLoop` constructor instantiates a
  :class:`~pyop2.base.LazyComputation` object: start halo exchange, compute
  core elements, finish halo exchange, compute owned elements, compute halo
  elements. This avoids a circular dependency problem in the code.

* Helper class (CORE, OWNED, HALO) help create finer dependencies for PyOP2
  Data objects.

* Instead of enqueuing halo exchange computation at the end of the trace, they
  are push as far back as possible (as long as the previous computation is
  independant of the halo echange).
