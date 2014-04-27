from pyop2 import op2, utils
import numpy as np

# RESULT: Each node should have a value 10 higher than its index.

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

# Constant declarations
op2.Const(6, [[0, 1], [9, 2], [0, 3]], 'k', dtype=np.int)
op2.Const(1, 5, 'j', dtype=np.int)

# Globals
p_global = op2.Global(2, data=[0, 2], name="p_global", dtype=np.int)

# Create kernel
kernel = op2.Kernel("""
void add_values(int *x, int *p_global) {
    (*x) = (*x) + j + k[5] + p_global[1];
}
""", "add_values")

# Print values before kernel invocation
print "--- Values before running kernel ---"
for i in range(0, nnode):
    print "Value at node", i, p_nodes.data[i]

# Execute kernel on each node
op2.par_loop(kernel, nodes, p_nodes(op2.RW), p_global(op2.READ))

# Print values after kernel invocation
print "--- Values after running kernel ---"
for i in range(0, nnode):
    print "Value at node", i, p_nodes.data[i]
