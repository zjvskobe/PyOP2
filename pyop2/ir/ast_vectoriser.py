from math import ceil
from copy import deepcopy as dcopy

from pyop2.ir.ast_base import *


class LoopVectoriser(object):

    """ Loop vectorizer
        * Vectorization:
          Outer-product vectorisation.
        * Memory:
          padding, data alignment, trip count/bound adjustment
          padding and data alignment are for aligned unit-stride load
          trip count/bound adjustment is for auto-vectorisation. """

    def __init__(self, loop_optimiser, isa, compiler):
        self.lo = loop_optimiser
        self.intr = self._set_isa(isa)
        self.comp = self._set_compiler(compiler)
        self.i_loops = []
        self._inner_loops(self.lo.loop_nest, self.i_loops)

    # Memory optimisations #

    def pad_and_align(self, decl):
        """Pad each data structure accessed in the loop nest to a multiple
        of the vector length. """

        # Get the declarations of the symbols accessed in the loop nest
        acc_decl = [d for s, d in decl.items() if s in self.lo.sym]

        # Do the rounding of the inner dimension (padding)
        for d in acc_decl:
            rounded = self._roundup(d.sym.rank[-1])
            d.sym.rank = d.sym.rank[:-1] + (rounded,)

    def adjust_loop(self, only_bound):
        """Adjust trip count and bound of each innermost loop in the nest."""

        # save a copy of the old loops
        self.orig_i_loops = dcopy(self.i_loops)

        for l in self.i_loops:
            # Also adjust the loop increment factor
            if not only_bound:
                l.incr.children[1] = c_sym(self.intr["dp_reg"])

            # Adjust the loops bound
            bound = l.cond.children[1]
            l.cond.children[1] = c_sym(self._roundup(bound.symbol))

    def set_alignment(self, decl, autovectorisation=False):
        """Align arrays in the kernel to the size of the vector length in
        order to issue aligned loads and stores. Also tell this information to
        the back-end compiler by adding suitable pragmas over loops in case
        we rely on autovectorisation. """

        for d in decl.values():
            d.attr.append(self.comp["align"](self.intr["alignment"]))

        if autovectorisation:
            for l in self.i_loops:
                l.pragma = self.comp["decl_aligned_for"]

    # Vectorisation

    def outer_product(self, opts=0):
        """Compute outer products according to opts.
            opts = 1 : peeling for autovectorisation
            opts = 2 : no peeling, just use padding
            opts = 3 : set unroll_and_jam factor
        """

        for stmt, stmt_info in self.lo.out_prods.items():
            # First, find outer product loops in the nest
            it_vars, parent = stmt_info
            loops = [l for l in self.lo.fors if l.it_var() in it_vars]

            vect_len = self.intr["dp_reg"]
            rows = loops[0].size()

            # Vectorisation
            op = OuterProduct(stmt, loops, self.intr, self.lo)
            extra_its = rows - vect_len
            if opts == 3:
                body, layout = op.generate(rows)
            elif extra_its >= 0:
                body, layout = op.generate(vect_len)
                if extra_its > 0 and opts in [1, 2]:
                # peel out
                    loop_peel = dcopy(loops)
                    # Adjust main, layout and remainder loops bound and trip
                    bound = loops[0].cond.children[1].symbol
                    bound -= bound % vect_len
                    loops[0].cond.children[1] = c_sym(bound)
                    layout.cond.children[1] = c_sym(bound)
                    loop_peel[0].init.init = c_sym(bound)
                    loop_peel[0].incr.children[1] = c_sym(1)
                    loop_peel[1].incr.children[1] = c_sym(1)
                    # Append peeling loop after the main loop
                    parent_loop = self.lo.fors[0]
                    for l in self.lo.fors[1:]:
                        if l.it_var() == loops[0].it_var():
                            break
                        else:
                            parent_loop = l
                    parent_loop.children[0].children.append(loop_peel[0])
            else:
                body, layout = op.generate(rows)

            # Insert the vectorized code at the right point in the loop nest
            blk = parent.children
            ofs = blk.index(stmt)
            parent.children = blk[:ofs] + body + blk[ofs + 1:]

            # Append the layout code after the loop nest
            if layout:
                parent = self.lo.pre_header.children
                parent.insert(parent.index(self.lo.loop_nest) + 1, layout)

    # Utilities
    def _inner_loops(self, node, loops):
        """Find the inner loops in the tree rooted in node."""
        if perf_stmt(node):
            return False
        elif isinstance(node, Block):
            return any([self._inner_loops(s, loops) for s in node.children])
        elif isinstance(node, For):
            found = self._inner_loops(node.children[0], loops)
            if not found:
                loops.append(node)
            return True

    def _set_isa(self, isa):
        """Set the proper intrinsics instruction set. """

        if isa == "AVX":
            return {
                "inst_set": "AVX",
                "avail_reg": 16,
                "alignment": 32,
                "dp_reg": 4,  # Number of double values per register
                "reg": lambda n: "ymm%s" % n,
                "zeroall": "_mm256_zeroall ()",
                "setzero": AVXSetZero(),
                "decl_var": "__m256d",
                "align_array": lambda p: "__attribute__((aligned(%s)))" % p,
                "symbol": lambda s, r: AVXLoad(s, r),
                "store": lambda m, r: AVXStore(m, r),
                "mul": lambda r1, r2: AVXProd(r1, r2),
                "div": lambda r1, r2: AVXDiv(r1, r2),
                "add": lambda r1, r2: AVXSum(r1, r2),
                "sub": lambda r1, r2: AVXSub(r1, r2),
                "l_perm": lambda r, f: AVXLocalPermute(r, f),
                "g_perm": lambda r1, r2, f: AVXGlobalPermute(r1, r2, f),
                "unpck_hi": lambda r1, r2: AVXUnpackHi(r1, r2),
                "unpck_lo": lambda r1, r2: AVXUnpackLo(r1, r2)
            }

    def _set_compiler(self, compiler):
        """Set compiler-specific keywords. """

        if compiler == "INTEL":
            return {
                "align": lambda o: "__attribute__((aligned(%s)))" % o,
                "decl_aligned_for": "#pragma vector aligned"
            }

    def _roundup(self, x):
        """Return x rounded up to the vector length. """
        word_len = self.intr["dp_reg"]
        return int(ceil(x / float(word_len))) * word_len


class OuterProduct(object):

    """Compute outer product vectorisation of a statement. """

    def __init__(self, stmt, loops, intr, nest):
        self.stmt = stmt
        self.intr = intr
        # Outer product loops
        self.loops = loops
        # The whole loop nest in which outer product loops live
        self.nest = nest

    class Alloc(object):

        """Handle allocation of register variables. """

        def __init__(self, intr, tensor_size):
            nres = max(intr["dp_reg"], tensor_size)
            self.ntot = intr["avail_reg"]
            self.res = [intr["reg"](v) for v in range(nres)]
            self.var = [intr["reg"](v) for v in range(nres, self.ntot)]
            self.i = intr

        def get_reg(self):
            if len(self.var) == 0:
                l = self.ntot * 2
                self.var += [self.i["reg"](v) for v in range(self.ntot, l)]
                self.ntot = l
            return self.var.pop(0)

        def free_regs(self, regs):
            for r in reversed(regs):
                self.var.insert(0, r)

        def get_tensor(self):
            return self.res

    def _swap_reg(self, step, vrs):
        """Swap values in a vector register. """

        # Find inner variables
        regs = [reg for node, reg in vrs.items()
                if node.rank and node.rank[-1] == self.loops[1].it_var()]

        if step in [0, 2]:
            return [self.intr["l_perm"](r, "5") for r in regs]
        elif step == 1:
            return [self.intr["g_perm"](r, r, "1") for r in regs]
        elif step == 3:
            return []

    def _vect_mem(self, node, vrs, decls):
        """Return a list of vector variables declarations representing
        loads, sets, broadcasts. Also return dicts of allocated inner
        and outer variables. """
        stmt = []
        for node, reg in vrs.items():
            exp = self.intr["symbol"](node.symbol, node.rank)
            if not decls.get(node.gencode()):
                decls[node.gencode()] = reg
                stmt.append(Decl(self.intr["decl_var"], reg, exp))
        return stmt

        return (decls, in_vrs, out_vrs)

    def _vect_expr(self, node, ofs, regs, decls, vrs={}):
        """Turn a scalar expression into its intrinsics equivalent.
        Also return dicts of allocated vector variables. """

        if isinstance(node, Symbol):
            if node.rank and self.loops[0].it_var() == node.rank[-1]:
                # The symbol depends on the outer loop dimension, so add ofst
                new_rank = list(node.rank)
                new_rank[-1] = new_rank[-1] + "+" + str(ofs)
                node = Symbol(node.symbol, tuple(new_rank))
            if node.gencode() not in decls:
                reg = regs.get_reg()
                vrs[node] = Symbol(reg, ())
                return (vrs[node], vrs)
            else:
                return (decls[node.gencode()], vrs)
        elif isinstance(node, Par):
            return self._vect_expr(node.children[0], ofs, regs, decls, vrs)
        else:
            left, vrs = self._vect_expr(
                node.children[0], ofs, regs, decls, vrs)
            right, vrs = self._vect_expr(
                node.children[1], ofs, regs, decls, vrs)
            if isinstance(node, Sum):
                return (self.intr["add"](left, right), vrs)
            elif isinstance(node, Sub):
                return (self.intr["sub"](left, right), vrs)
            elif isinstance(node, Prod):
                return (self.intr["mul"](left, right), vrs)
            elif isinstance(node, Div):
                return (self.intr["div"](left, right), vrs)

    def _incr_tensor(self, tensor, ofs, regs, out_reg, mode):
        """Add the right hand side contained in out_reg to tensor."""
        if mode == 0:
            # Store in memory
            loc = (tensor.rank[0] + "+" + str(ofs), tensor.rank[1])
            return self.intr["store"](Symbol(tensor.symbol, loc), out_reg)
        elif mode == 1:
            # Accumulate on a vector register
            reg = Symbol(regs.get_tensor()[ofs], ())
            return Assign(reg, self.intr["add"](reg, out_reg))

    def _restore_layout(self, regs, tensor, mode):
        """Restore the storage layout of the tensor. """

        code = []
        t_regs = [Symbol(r, ()) for r in regs.get_tensor()]
        n_regs = len(t_regs)

        # Determine tensor symbols
        tensor_syms = []
        for i in range(n_regs):
            rank = (tensor.rank[0] + "+" + str(i), tensor.rank[1])
            tensor_syms.append(Symbol(tensor.symbol, rank))

        # Load LHS values from memory
        if mode == 0:
            for i, j in zip(tensor_syms, t_regs):
                load_sym = self.intr["symbol"](i.symbol, i.rank)
                code.append(Decl(self.intr["decl_var"], j, load_sym))

        # In-register restoration of the tensor
        # TODO: AVX only at the present moment
        # TODO: here some __m256 vars could not be declared if rows < 4
        perm = self.intr["g_perm"]
        uphi = self.intr["unpck_hi"]
        uplo = self.intr["unpck_lo"]
        typ = self.intr["decl_var"]
        vect_len = self.intr["dp_reg"]
        spins = int(ceil(n_regs / float(vect_len)))
        for i in range(spins):
            # In-register permutations
            tmp = [Symbol(regs.get_reg(), ()) for r in range(vect_len)]
            code.append(Decl(typ, tmp[0], uphi(t_regs[1], t_regs[0])))
            code.append(Decl(typ, tmp[1], uplo(t_regs[0], t_regs[1])))
            code.append(Decl(typ, tmp[2], uphi(t_regs[2], t_regs[3])))
            code.append(Decl(typ, tmp[3], uplo(t_regs[3], t_regs[2])))
            code.append(Assign(t_regs[0], perm(tmp[1], tmp[3], 32)))
            code.append(Assign(t_regs[1], perm(tmp[0], tmp[2], 32)))
            code.append(Assign(t_regs[2], perm(tmp[3], tmp[1], 49)))
            code.append(Assign(t_regs[3], perm(tmp[2], tmp[0], 49)))
            regs.free_regs([s.symbol for s in tmp])

            # Store LHS values in memory
            for j in range(min(vect_len, n_regs - i * vect_len)):
                ofs = i * vect_len + j
                code.append(self.intr["store"](tensor_syms[ofs], t_regs[ofs]))

        return code

    def generate(self, rows):
        """Generate the outer-product intrinsics-based vectorisation code. """

        cols = self.intr["dp_reg"]

        # Determine order of loops w.r.t. the local tensor entries.
        # If j-k are the inner loops and A[j][k], then increments of
        # A are performed within the k loop, otherwise we would lose too many
        # vector registers for keeping tmp values. On the other hand, if i is
        # the innermost loop (i.e. loop nest is j-k-i), stores in memory are
        # done outside of ip, i.e. immediately before the outer product's
        # inner loop terminates.
        if self.loops[1].it_var() == self.nest.fors[-1].it_var():
            # Stores
            mode = 0
            tensor_size = cols
        else:
            # Local increments
            mode = 1
            tensor_size = rows

        tensor = self.stmt.children[0]
        expr = self.stmt.children[1]

        # Get source-level variables
        regs = self.Alloc(self.intr, tensor_size)

        # Adjust loops increment
        self.loops[0].incr.children[1] = c_sym(rows)
        self.loops[1].incr.children[1] = c_sym(cols)

        stmt = []
        decls = {}
        rows_per_col = rows / cols
        rows_to_peel = rows % cols
        peeling = 0
        for i in range(cols):
            # Handle extra rows
            if peeling < rows_to_peel:
                nrows = rows_per_col + 1
                peeling += 1
            else:
                nrows = rows_per_col
            for j in range(nrows):
                # Vectorize, declare allocated variables, increment tensor
                ofs = j * cols
                v_expr, vrs = self._vect_expr(expr, ofs, regs, decls)
                stmt.extend(self._vect_mem(expr, vrs, decls))
                incr = self._incr_tensor(tensor, i + ofs, regs, v_expr, mode)
                stmt.append(incr)
            # Register shuffles
            if rows_per_col + (rows_to_peel - peeling) > 0:
                stmt.extend(self._swap_reg(i, vrs))

        # Set initialising and tensor layout code
        layout = self._restore_layout(regs, tensor, mode)
        if mode == 0:
            # Tensor layout
            layout_loops = dcopy(self.loops)
            layout_loops[0].incr.children[1] = c_sym(cols)
            layout_loops[0].children = [
                Block([layout_loops[1]], open_scope=True)]
            layout_loops[1].children = [Block(layout, open_scope=True)]
            layout = layout_loops[0]
        elif mode == 1:
            # Initialiser
            for r in regs.get_tensor():
                decl = Decl(self.intr["decl_var"], Symbol(r, ()),
                            self.intr["setzero"])
                self.loops[1].children[0].children.insert(0, decl)
            # Tensor layout
            self.loops[1].children[0].children.extend(layout)
            layout = None

        return (stmt, layout)