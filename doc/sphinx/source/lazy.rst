Lazy Evaluation
===============

Approach
--------

Unless explicitly set up to execute greedily, PyOP2 implements a lazy
evaluation scheme where steps of PyOP2's computations are delayed for later
execution. This scheme opens up a range of optimisations and task rescheduling.

An example
----------

In this document, we examine this lazy evaluation scheme through an example
from the Firedrake_ test suite, an explicit wave simulation.

.. code-block:: python

    from firedrake import *

    output = True

    mesh = UnitSquareMesh(100, 100)

    T = 10
    dt = 0.001
    t = 0
    fs = FunctionSpace(mesh, 'Lagrange', 1)
    p = Function(fs)
    phi = Function(fs)

    u = TrialFunction(fs)
    v = TestFunction(fs)

    p.interpolate(Expression("exp(-40*((x[0]-.5)*(x[0]-.5)+(x[1]-.5)*(x[1]-.5)))"))

    outfile = File("out.pvd")
    phifile = File("phi.pvd")

    outfile << p
    phifile << phi
    
    step = 0
    while t <= T:
        step += 1

        phi -= dt / 2 * p

        p += (assemble(dt * inner(nabla_grad(v), nabla_grad(phi)) * dx)
              / assemble(v * dx))

        phi -= dt / 2 * p

        t += dt

        outfile << p
        phifile << phi

In this example, we will focus on the execution of the main loop. This code
translates to seven :func:`~pyop2.par_loop` calls executed by PyOP2:

.. code-block:: python

    while t <- T:
        step += 1

        par_loop(phi_expr, ..., p(READ), phi(INC))
        par_loop(zero, ..., t1(WRITE))
        par_loop(assemble1, ..., v(READ, map), phi(READ, map), t1(WRITE))
        par_loop(zero, ..., t2(WRITE))
        par_loop(assemble2, ..., v(READ, map), t2(WRITE))
        par_loop(expr, ..., t1(READ), t2(READ), p(WRITE))
        par_loop(phi_expr, ..., p(READ), phi(INC))

        outfile << p.data
        outfile << phi.data

All loops excepts for the two assembly calls are direct loops.

Delayed computations
---------------------

In order to delay the execution of parallel loops, :func:`~pyop2.par_loop`
calls do not perform any computations right away. Instead, PyOP2 instantiates
:class:`~pyop2.lazy.LazyComputation` objects. These are *closures* whose
:meth:`_run` method will perform the actual computation. PyOP2 implements
specialised versions of :class:`~pyop2.lazy.LazyComputation` for each delayed
computation:

* :class:`~pyop2.lazy.LazyCompute`, performs the actual kernel execution over
  a part of the iteration space, which has four sections, namely `core`,
  `owned`, `halo_exec` and `halo_import`, as described in the :doc:`mpi`
  documentation.

* :class:`~pyop2.lazy.LazyHaloSend` and :class:`~pyop2.lazy.LazyHaloRecv` deal
  with the synchronisation of mesh data across MPI processes.

* :class:`~pyop2.lazy.LazyReductionBegin` and
  :class:`~pyop2.lazy.LazyReductionEnd` perform the reduction of
  :class:`Globals <pyop2.Global>` across MPI processes.

Back to our example, the first :func:`~pyop2.par_loop` call would instantiate
two :class:`~pyop2.lazy.LazyCompute` objects, one for the `core` and one for
the `owned` part of its iteration space. The third call which is an indirect
loop, would create six :class:`~pyop2.lazy.LazyComputation`: two
:class:`~pyop2.lazy.LazyHaloSend` for each Dat read (``v`` and ``phi``), one
:class:`~pyop2.lazy.LazyCompute` over the `core` elements of the iteration
set, two :class:`~pyop2.lazy.LazyHaloRecv` matching the previous send, and
finally, another :class:`~pyop2.lazy.LazyCompute` for the `owned` elements.

Declaring computations
----------------------

For the purpose of reordering the execution of computations while preserving
the correctness of the PyOP2 program, instances of
:class:`~pyop2.lazy.LazyComputation` must declare their input and output
dependencies, that is the set of :class:`~pyop2.Dat` objects that is read
and/or written during the execution of the
:class:`~pyop2.lazy.LazyComputation`. This information is used by PyOP2 to
maintain the correct data dependencies when reordering computations.

A :class:`~pyop2.lazy.LazyComputation`'s reads and writes are directly derived
from the arguments of the :class:`~pyop2.base.ParLoop` through the instance
attributes :attr:`reads` and :attr:`writes` of
:class:`~pyop2.lazy.LazyComputation`. Since closures do not necessarily read
and write entire sections of Dats, PyOP2 further refines dependencies as
sections of Dats (see :class:`~pyop2.lazy.CORE`, :class:`~pyop2.lazy.OWNED`,
:class:`~pyop2.lazy.HALOEXEC` and :class:`pyop2.lazy.HALOIMPORT`).

Applied to the example above:

* The iteration over the `core` elements of the first :func:`~pyop2.par_loop`
  reads ``CORE(phi)`` and ``CORE(p)``, the core elements of ``phi`` and ``p``.

* The iteration over the `owned` elements of the first indirect loop, access
  only the currently iterated part of direct :class:`Dats <pyop2.Dat>`, while
  in the case of indirect :class:`Dats <pyop2.Dat>`, neighbouring sections are
  also accessed. Thus, the reads set is ``CORE(phi)``, ``OWNED(phi)``,
  ``HALOEXEC(phi)``, ``CORE(v)``, ``OWNED(v)``, and ``HALOEXEC(v)`` since both
  ``phi`` and ``v`` are accessed indirectly. The writes set contains only
  ``OWNED(t1)`` which is accessed directly.

Delaying computations
---------------------

In order to maintain correct data dependencies between
:class:`~pyop2.lazy.LazyComputation`\s, these are stored in an instance of
:class:`~pyop2.lazy.ExecutionTrace` which maintain a directed acyclic graph of
:class:`~pyop2.lazy.LazyComputation`\s, which is a partial order on the
delayed computations.

In order to correctly insert a :class:`~pyop2.lazy.LazyComputation` into the
DAG, lazy computation objects implement a
:meth:`~pyop2.lazy.LazyComputation.depends_on` method which tests if one
computations depends on the execution of another closure. This method is
described below.

Let us call :math:`R_a`, :math:`W_a`, :math:`R_c` and :math:`W_c`, the read
(:math:`R`) and write (:math:`W`) dependencies of passed in arguments
(:math:`a`) and of the delayed computation (:math:`c`) respectively.
Computation :math:`c` is required to be executed if

.. math::

    R_a \cap W_c \cup W_a \cap R_c \cup W_a \cap W_c

is not empty. In other words, if a dependency read is writen by :math:`c`, if
a dependency written must be read first, or if a dependency is being
overwritten, preserving write ordering.

Back to our example, before any computation is enqueued, our DAG is empty.
The first :func:`pyop2.par_loop` call creates one
:class:`~pyop2.lazy.LazyCompute` for the iteration over `core` and `owned`
element.

.. graphviz::

    digraph a {
        top [label="top"];
        c1 [label="phi_expr[core,owned]"];
        bot [label="bot"];

        top -> c1 -> bot;
    }

Similarly, the second :func:`~pyop2.par_loop` is zeroing a temporary
:class:`~pyop2.Dat`, because this does not depend on the previous
:func:`~pyop2.par_loop` execution, the second :class:`~pyop2.lazy.LazyCompute`
is a sibling of the previous one:

.. graphviz::

    digraph b {
        top [label="top"];
        c1 [label="phi_expr[core,owned]"];
        c2 [label="zero(t1)[core,owned]"];
        bot [label="bot"];

        top -> c1 -> bot;
        top -> c2 -> bot;
    }

Next, the assemble will instantiate the following
:class:`~pyop2.lazy.LazyComputation` with reads and writes, as follows:

================  ==========================   ==============================
Computation       Reads                        Writes
================  ==========================   ==============================
HaloSend(phi)     CORE(phi), OWNED(phi)        NET(phi)
----------------  --------------------------   ------------------------------
HaloSend(v)       CORE(v), OWNED(v)            NET(v)
----------------  --------------------------   ------------------------------
assemble1[CORE]   CORE(phi), CORE(v)           CORE(t1)
                  OWNED(phi), OWNED(v)
----------------  --------------------------   ------------------------------
HaloRecv(phi)     NET(phi)                     HALOEXEC(phi), HALOIMPORT(phi)
----------------  --------------------------   ------------------------------
HaloRecv(v)       NET(v)                       HALOEXEC(v), HALOIMPORT(v)
----------------  --------------------------   ------------------------------
assemble1[OWNED]  CORE(phi), CORE(v)           OWNED(t1)
                  OWNED(phi), OWNED(v)
                  HALOEXEC(phi), HALOEXEC(v)
================  ==========================   ==============================


As the first halo send must read ``phi``, it depends on the execution of the
first :func:`~pyop2.par_loop`. Therefore, the
:class:`~pyop2.lazy.LazyHaloSend` will be a child of the ``phi_expr`` compute.
On the other hand, the second halo send will be a new parallel branch as no
other computation deals with ``v``. As the assembly over the core elements
depends on ``v``, ``phi``, and ``t1``, as the first halo send, it is a child
of the first :func:`~pyop2.par_loop`, and it is a child of the
:func:`~pyop2.par_loop` zeroing ``t1``. The next two ``HaloRecv`` do not
depend on the previous computation since all of them only read ``phi`` and
``v`` but depend on their matching ``HaloSend`` via the ``NET(phi)`` and
``NET(v)``, representing an abstract communication stub to enforce the
ordering between sends and receives.  Finally, the computation over the
`owned` elements must occur after the computation on the `core` ones and after
the two ``HaloRecv``, giving the DAG below:

.. graphviz::

    digraph b {
        top [label="top"];
        c1 [label="phi_expr[core,owned]"];
        c2 [label="zero(t1)[core,owned]"];
        c3 [label="HaloSend(phi)"];
        c4 [label="HaloSend(v)"];
        c5 [label="assemble1[core]"];
        c6 [label="HaloRecv(phi)"];
        c7 [label="HaloRecv(v)"];
        c8 [label="assemble1[owned]"];
        bot [label="bot"];

        top -> c1;
        top -> c2;
        top -> c4;

        c1 -> c3;
        c1 -> c5;

        c2 -> c5;
        c3 -> c6;
        c4 -> c7;

        c5 -> c8;
        c6 -> c8;
        c7 -> c8;

        c8 -> bot;
    }

The calls to the other :func:`~pyop2.par_loop` will produce the following
partial order:

.. graphviz::

        digraph finale {
            A1a2 [label="HaloSend(v)" ];
            P [label="p_expr[core,owned]" ];
            A1b [label="assemble1[core]" ];
            A2c2 [label="HaloRecv(v)" ];
            Z2 [label="zero(t2)" ];
            A2d [label="assemble2[owned]" ];
            A2b [label="assemble2[core]" ];
            A2a1 [label="HaloSend(phi)" ];
            DT2 [label="phi_expr[core,owned]" ];
            A2c1 [label="HaloRecv(phi)" ];
            DT1 [label="phi_expr[core,owned]" ];
            A1c2 [label="HaloRecv(v)" ];
            top [label="top" ];
            bot [label="bot" ];
            Z1 [label="zero(t1)" ];
            A1a1 [label="HaloSend(phi)" ];
            A1d [label="assemble1[owned]" ];
            A1c1 [label="HaloRecv(phi)" ];
            A2a2 [label="HaloSend(v)" ];


            top -> Z2 -> A2b -> A2d;
            DT1 -> A2b;
            top -> DT1 -> A1a1 -> A1c1 -> A2a1 -> A2c1 -> A2d;
            DT1 -> A1b;

            top-> A1a2 -> A1c2 -> A2a2 -> A2c2 -> A2d;
            top -> Z1 -> A1b -> A1d;
            A1d -> A2c2;
            A1d -> A2c1;
            A1c2 -> A1d;

            A2d -> P -> DT2 -> bot;
        }

Forcing computations
--------------------

In PyOP2's lazy evaluation scheme, computations are delayed until the content
of a :class:`~pyop2.base.DataCarrier` is examined through the user exposed
methods. At that point, the :class:`~pyop2.lazy.ExecutionTrace` must decide on
an execution plan based on the partial order provided by the directed acyclic
graph. At this point, the :class:`~pyop2.lazy.ExecutionTrace` recursively
descends the DAG applying transformation and choosing the "best" scheduling.

One such optimisation is the reordering of halo exchanges to benefit from
communication/computation overlaps. This will systematically schedule halo
sends first whenever an alternative is encountered.

Once this rewriting phase is performed, the total order on the
:class:`~pyop2.lazy.LazyComputation` is executed and the
:class:`~pyop2.lazy.ExecutionTrace` is emptied.

Future Work
-----------

* Early halo skips: pairs of halo exchanges for which the `send` can be
  scheduled immediately (that is the `send` is a child of the `top` node) can
  be removed from the DAG if the halo of that :class:`~pyop2.Dat` is up to
  date. This is not per se an optimisation, as no actual exchange would happen
  there, but a simplification of the DAG:

.. graphviz::

        digraph finale {
            A1a2 [label="HaloSend(v)" color=red fontcolor=red];
            P [label="p_expr[core,owned]" ];
            A1b [label="assemble1[core]" ];
            A2c2 [label="HaloRecv(v)"];
            Z2 [label="zero(t2)" ];
            A2d [label="assemble2[owned]" ];
            A2b [label="assemble2[core]" ];
            A2a1 [label="HaloSend(phi)" ];
            DT2 [label="phi_expr[core,owned]" ];
            A2c1 [label="HaloRecv(phi)" ];
            DT1 [label="phi_expr[core,owned]" ];
            A1c2 [label="HaloRecv(v)" color=red fontcolor=red];
            top [label="top" ];
            bot [label="bot" ];
            Z1 [label="zero(t1)" ];
            A1a1 [label="HaloSend(phi)" ];
            A1d [label="assemble1[owned]" ];
            A1c1 [label="HaloRecv(phi)" ];
            A2a2 [label="HaloSend(v)"];
            
            
            top -> Z2 -> A2b -> A2d;
            DT1 -> A2b;
            top -> DT1 -> A1a1 -> A1c1 -> A2a1 -> A2c1 -> A2d;
            DT1 -> A1b;
            
            top-> A1a2 -> A1c2 -> A2a2 -> A2c2 -> A2d;
            top -> Z1 -> A1b -> A1d;
            A1d -> A2c2;
            A1d -> A2c1;
            A1c2 -> A1d;
            
            A2d -> P -> DT2 -> bot;
        }

* Successive halo exchanges: On two successive halo exchanges, the second one
  can be systematically removed (since the halo will be up to date, whenever a
  ``halo_send`` is directly dependent on a ``halo_recv``), again, this is not
  an optimisation per se, but a graph simplification:

.. graphviz::

        digraph finale {
            A1a2 [label="HaloSend(v)" color=blue fontcolor=blue];
            P [label="p_expr[core,owned]" ];
            A1b [label="assemble1[core]" ];
            A2c2 [label="HaloRecv(v)" color=red fontcolor=red];
            Z2 [label="zero(t2)" ];
            A2d [label="assemble2[owned]" ];
            A2b [label="assemble2[core]" ];
            A2a1 [label="HaloSend(phi)" ];
            DT2 [label="phi_expr[core,owned]" ];
            A2c1 [label="HaloRecv(phi)" ];
            DT1 [label="phi_expr[core,owned]" ];
            A1c2 [label="HaloRecv(v)" color=blue fontcolor=blue];
            top [label="top" ];
            bot [label="bot" ];
            Z1 [label="zero(t1)" ];
            A1a1 [label="HaloSend(phi)" ];
            A1d [label="assemble1[owned]" ];
            A1c1 [label="HaloRecv(phi)" ];
            A2a2 [label="HaloSend(v)" color=red fontcolor=red];
            
            
            top -> Z2 -> A2b -> A2d;
            DT1 -> A2b;
            top -> DT1 -> A1a1 -> A1c1 -> A2a1 -> A2c1 -> A2d;
            DT1 -> A1b;
            
            top-> A1a2 -> A1c2 -> A2a2 -> A2c2 -> A2d;
            top -> Z1 -> A1b -> A1d;
            A1d -> A2c2;
            A1d -> A2c1;
            A1c2 -> A1d;
            
            A2d -> P -> DT2 -> bot;
        }

* Soft kernel loop fusion: by definition, sibling compute operations can be
  fused simply, provided they have the same iteration set. One good candidate
  for this would be parallel loops zeroing temporary dats.

* Soft kernel loop fusion of direct loops: two directly dependent compute
  operations on the same iteration set can also be fused if the parallel loops
  are direct.

* Exposing more PyOP2 operations to the lazy evaluation scheme: currently,
  temporary :class:`Dats <pyop2.Dat>` (:class:`Dats <pyop2.Dat>` created from
  a zeroed memory) are created in PyOP2 outside of the lazy evaluation scheme.
  The only visible part of the temporary :class:`~pyop2.Dat` creation is the
  zeroing :func:`~pyop2.par_loop`. By exposing each step of memory allocation
  (allocate memory and release memory) this would open up the opportunity to
  effectively reasign memory to new temporaries. In our running example, two
  temporaries are created for the two assembly operations. Both are zeroed
  before the actual assembly loops, and are destroyed after the ``p += t1 /
  t2`` loop.  Provided that the lifecycle of a temporary creates the following
  closures:

  =================  =====  =====================
  closure            reads  writes
  =================  =====  =====================
  LazyMalloc(x)             MEM
                            CORE(dat), OWNED(dat)
  -----------------  -----  ---------------------
  zero[CORE, OWNED]         CORE(dat), OWNED(dat)
  -----------------  -----  ---------------------
  LazyFree(x)               MEM
                            CORE(dat), OWNED(dat)
  =================  =====  =====================

  Then two successive iterations of our example, would produce the following
  graph (provided the above halo exchange optimisations are implemented):

.. graphviz::

        digraph hypothetical {
            P [label="p_expr[core,owned]" ];
            A1b [label="assemble1[core]" ];
            Z2 [label="zero(t2)" ];
            A2d [label="assemble2[owned]" ];
            A2b [label="assemble2[core]" ];
            DT2 [label="phi_expr[core,owned]" ];
            DT1 [label="phi_expr[core,owned]" ];
            top [label="top" ];
            bot [label="bot" ];
            T1 [label="malloc(t1)" color=red fontcolor=red];
            T2 [label="malloc(t2)" color=red fontcolor=red];
            F1 [label="free(t1)" color=red fontcolor=red];
            F2 [label="free(t2)" color=red fontcolor=red];
            Z1 [label="zero(t1)" ];
            A1a1 [label="HaloSend(phi)" ];
            A1d [label="assemble1[owned]" ];
            A1c1 [label="HaloRecv(phi)" ];

            { rank=min; top; }
            { rank=max; bot; }

            top -> T2 -> Z2 -> A2b -> A2d;
            DT1 -> A2b;
            top -> DT1 -> A1a1 -> A1c1 -> A2d;
            DT1 -> A1b;

            top -> T1 -> Z1 -> A1b -> A1d;
            A1d -> A2d;

            A2d -> P;
            P -> DT2 -> bot;
            P -> F1 -> bot;
            P -> F2 -> bot;

           bot -> top;
        }

This in turn, would make it possible to "reuse" memory allocated for temporary
dats accross successive iteration, and fusing the zeroing, the ``p_expr`` and
``phi_expr`` loops into one.

.. _Firedrake: http://firedrakeproject.org
