from pyop2 import op2, utils
import numpy as np

# Increments the value of an array using LLVM backend.

parser = utils.parser(group=True, description=__doc__)

opt = vars(parser.parse_args())
opt["backend"] = "sequential_llvm"
op2.init(**opt)

# Create sample data
A = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# Create set of nodes.
nnode = 10
nodes = op2.Set(nnode, "nodes")
p_nodes = op2.Dat(nodes, data=A, name="p_nodes")

# Create kernel
kernel_opts = {'llvm_kernel': True}
kernel = op2.Kernel("""
define void @direct_inc(i32* %x) nounwind {
  %1 = alloca i32*, align 4
  store i32* %x, i32** %1, align 4
  %2 = load i32** %1, align 4
  %3 = load i32* %2, align 4
  %4 = add nsw i32 %3, 1
  %5 = load i32** %1, align 4
  store i32 %4, i32* %5, align 4
  ret void
}
""", "direct_inc", kernel_opts)

# Print values before kernel invocation
print "--- Values before running kernel ---"
for i in range(0, nnode):
    print "Value at node", i, p_nodes.data[i]

# Execute kernel on each node
op2.par_loop(kernel, nodes, p_nodes(op2.RW))

# Print values after kernel invocation
print "--- Values after running kernel ---"
for i in range(0, nnode):
    print "Value at node", i, p_nodes.data[i]
