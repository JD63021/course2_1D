#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# ---------------- user parameters --------------------------------------
r_inner   = 1.0
r_outer   = 12.0
phi_inner = 50.0
kappa     = 1.0
Nelem     = 100   # number of elements
# -----------------------------------------------------------------------

# uniform mesh and connectivity
r    = np.linspace(r_inner, r_outer, Nelem+1)
elem = np.vstack([np.arange(Nelem), np.arange(1, Nelem+1)]).T  # (Nelem×2)

nNodes = r.size
K      = np.zeros((nNodes, nNodes))
F      = np.zeros(nNodes)

# 2-point Gauss rule on [0,1] (mapped from ±1/√3 on [−1,1])
xi_ref = np.array([-1.0, 1.0]) / np.sqrt(3)
xiGP   = (xi_ref + 1.0) / 2.0
wGP    = np.array([0.5, 0.5])

def shapeLin(xi):
    """ Linear shape functions on [0,1] """
    N     = np.array([1-xi, xi])
    dNdxi = np.array([-1.0, 1.0])
    return N, dNdxi

def isomap(re, N, dNdxi):
    """ Isoparametric map from ξ∈[0,1] → physical r """
    drdxi = np.dot(dNdxi, re)   # dr/dξ
    r_iso = np.dot(N,    re)    # r(ξ)
    L     = drdxi              # Jacobian
    return r_iso, drdxi, L

# element loop
for e in range(Nelem):
    nodes = elem[e]
    ke    = np.zeros((2,2))
    re    = r[nodes]

    for g in range(2):
        xi, w    = xiGP[g], wGP[g]
        Nvals, dNdxi = shapeLin(xi)
        r_iso, drdxi, L = isomap(re, Nvals, dNdxi)
        dN       = dNdxi / drdxi   # dN/dr

        # spherical weighting: r^2
        ke += kappa * np.outer(dN, dN) * (r_iso**2) * L * w

    # assemble
    idx = np.ix_(nodes, nodes)
    K[idx] += ke

# apply boundary conditions
# Dirichlet at r_inner
K[0, :] = 0.0
K[0, 0] = 1.0
F[0]    = phi_inner

# Robin at r_outer: dφ/dr + (1/r_outer)*φ = 0
beta           = 1.0 / r_outer
K[-1, -1]     += beta * r_outer**2
# F[-1] stays zero (homogeneous)

# solve
phi_fem      = np.linalg.solve(K, F)
phi_analytic = phi_inner / r

# plot
plt.figure()
plt.plot(r, phi_fem,    'o-', label='FEM')
plt.plot(r, phi_analytic,'-',  label='Analytic')
plt.xlabel('r')
plt.ylabel(r'$\phi(r)$')
plt.title('1D Spherical FEM vs. Analytic: φ = φ_inner / r')
plt.legend()
plt.grid(True)
plt.show()

# error
print(f"Max |φ_FEM − φ_exact| = {np.max(np.abs(phi_fem - phi_analytic)):.3e}")
