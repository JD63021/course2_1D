# spherical_radial_poisson.jl
using LinearAlgebra
using Plots

# ---------------- user parameters --------------------------------------
r_inner   = 1.0
r_outer   = 12.0
phi_inner = 50.0
kappa     = 1.0
Nelem     = 100          # number of elements
# -----------------------------------------------------------------------

# uniform mesh and connectivity
r    = collect(range(r_inner, stop=r_outer, length=Nelem+1))
elem = [ [i, i+1] for i in 1:Nelem ]   # Vector of 2-element vectors

nNodes = length(r)
K      = zeros(nNodes, nNodes)
F      = zeros(nNodes)

# 2-point Gauss on [0,1], mapped from ±1/√3 on [−1,1]
xi_ref = [-1, 1] ./ sqrt(3)
xiGP   = (xi_ref .+ 1.0) ./ 2.0
wGP    = fill(0.5, 2)

# linear shape functions on [0,1]
function shapeLin(ξ)
    N     = [1-ξ, ξ]
    dNdxi = [-1.0, 1.0]
    return N, dNdxi
end

# isoparametric map from ξ → r
function isomap(re, N, dNdxi)
    drdxi = dot(dNdxi, re)   # dr/dξ
    r_iso = dot(N,    re)    # r(ξ)
    J     = drdxi            # Jacobian
    return r_iso, drdxi, J
end

# assemble K, F
for e in 1:Nelem
    nodes = elem[e]
    ke    = zeros(2,2)
    re    = r[nodes]

    for g in 1:2
        ξ      = xiGP[g]
        w      = wGP[g]
        Nvals, dNdxi = shapeLin(ξ)
        r_iso, drdxi, J = isomap(re, Nvals, dNdxi)
        dN    = dNdxi ./ drdxi         # dN/dr

        # spherical weighting: r^2
        ke .+= kappa * (dN * dN') * (r_iso^2) * J * w
    end

    K[nodes, nodes] .+= ke
end

# Dirichlet at r_inner
K[1, :] .= 0.0
K[1, 1]  = 1.0
F[1]     = phi_inner

# Robin at r_outer: dφ/dr + (1/r_outer)*φ = 0
beta = 1.0 / r_outer
K[end, end] += beta * r_outer^2

# solve
φ_fem   = K \ F
φ_exact = phi_inner ./ r

# build and display the plot
p1 = plot(r, φ_fem;
    marker=:circle, label="FEM",
    title="1D Spherical FEM vs Analytic: φ = φ_inner / r",
    xlabel="r", ylabel="φ(r)",
    grid=true)

plot!(p1, r, φ_exact; label="Analytic", lw=2)
display(p1)

# max error
max_err = maximum(abs.(φ_fem .- φ_exact))
@info "Max |φ_FEM - φ_exact| = $(round(max_err, sigdigits=5))"
