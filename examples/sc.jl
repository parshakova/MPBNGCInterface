# Supply chain example taken from OSBDO

using MPBNGCInterface
using Convex, ECOS, OSQP
using Pandas 

include("osbdo_funcs.jl")
# using .OSBDO

opt = BundleOptions( 
		OPT_RL => .01, # "Line search parameter (0 < RL < 0.5)."
		OPT_EPS => 1e-5, # "The final objective function accuracy parameter (0 < EPS)."
		OPT_FEAS => 1e-9, # "The tolerance for constraint feasibility (0 < FEAS)."
		OPT_IPRINT => 3, 
		OPT_JMAX => 10000,  # "The maximum number of stored subgradients (2 <=JMAX)."
		OPT_NOUT => 6, # "The output file number."
		OPT_NITER => 200, # "The maximum number of iterations (0 < NITER)."
		OPT_NFASG => 20000, # "The maximum number of FASG calls (1 < NFASG)."
		OPT_LMAX => 100) # "The maximum number of FASG calls in line search (0 < LMAX)."


data = read_pickle("/Users/parshakova.tanya/Documents/Distributed_Optimization/OSBDO_private/examples/supply_chain/sc_data.pickle")
params = data["params"]
lb = data["lower_bound"]
ub = data["upper_bound"]

size_x = size(lb)[1] # size of x

C = Matrix(transpose(data["A"]))
ubc = data["b"] # linear constraints contain conservation of flow 
lbc = data["b"]
grad_g_val = vec(data["grad_g_val"])
h_cvx = data["h_cvx"]


mu = 50.
norm = "l1"

# feasible starting value
x = init_feasible_point(size_x, params)

@show size(grad_g_val), size(x)

test_f_subgrad(params, size_x, norm, mu)
println("PASSED test_f_subgrad")
test_h_subgrad(params, size_x, norm, mu, grad_g_val)
println("PASSED test_h_subgrad")

function fasg(size_x, x, mm, f, g)

    subgrad, h_val = sc_hval_subgrad(params, size_x, x, norm, mu, grad_g_val)
	f[1] = h_val
	g[begin:end, 1] = subgrad

end

@show size(lb), size(ub), size(lbc), size(ubc), size(C)

prob = BundleProblem(size_x, fasg, x, vec(lb), vec(ub), vec(lbc), vec(ubc), C)

(x, fval, ierr, stats) = solveProblem(prob, opt)

display(stats); println()

@show fval[1]
# @show x
@show h_cvx

