%% spherical_radial_poisson_solver.m  ---------------------------------
% 1-D spherical Laplace (1/r^2) d/dr ( r^2 dφ/dr ) = 0
% Dirichlet  : φ(r_inner) = φ_inner
% Robin      : dφ/dr + (1/r_outer)*φ = 0   at r = r_outer
% Linear 2-node elements on a uniform mesh, 2-point Gauss on [0,1]

clear; close all; clc

% ---------------- user parameters --------------------------------------
r_inner   = 1.0;
r_outer   = 12.0;
phi_inner = 50.0;
kappa     = 1.0;
Nelem     = 100;               % number of elements
% -----------------------------------------------------------------------

%% uniform mesh and connectivity
r    = linspace(r_inner, r_outer, Nelem+1)';   % (Nelem+1) nodes
elem = [(1:Nelem)'  (2:Nelem+1)'];             % (Nelem × 2)
for e = 1:Nelem
    i0 = (e-1) + 1;
    elem(e,:) = [i0, i0+1];
end   

nNodes = numel(r);
K      = zeros(nNodes);       % global stiffness
F      = zeros(nNodes,1);     % global load (zero here)

%% 2-point Gauss rule on [0,1]
% map ξ_ref = ±1/√3 → ξ = (ξ_ref+1)/2
xiGP = ([-1, 1]/sqrt(3) + 1)/2;
wGP  = [0.5, 0.5];
nnode = numel(size(elem)); nnode = 2;

%% element loop
for e = 1:Nelem
    nodes = elem(e,:);
    ke    = zeros(nnode,nnode);      % element stiffness
    re    = r(nodes);
    
    % --- Gauss points
    for g = 1:2
        xi        = xiGP(g);
        w         = wGP(g);
        [N,dNdxi] = shapeLin(xi);           % size 2×1
        [r_iso,drdxi,L] = isomap(re, N, dNdxi);
        dN        = dNdxi / drdxi;         % dN/dr
        
        % spherical weighting: r^2
        ke = ke + kappa * (dN*dN.') * (r_iso^2) * L * w;
    end
    
    % assemble into global K
    K(nodes,nodes) = K(nodes,nodes) + ke;
end

%% apply BCs ------------------------------------------------------------
% Dirichlet at r_inner
K(1,:)   = 0;  
K(1,1)   = 1;   
F(1)     = phi_inner;

% Robin at r_outer: dφ/dr + (1/r_outer)*φ = 0
beta = 1 / r_outer;
% weak form adds β·r_outer^2 to K(end,end), zero to F(end)
K(end,end) = K(end,end) + beta * r_outer^2;

%% solve
phi_fem = K \ F;

%% analytic solution for comparison: φ = φ_inner / r
phi_analytic = phi_inner ./ r;

%% plots
figure
plot(r, phi_fem,    'o-'); hold on
plot(r, phi_analytic, '-');

fprintf('Max |φ_FEM − φ_exact| = %.3e\n', max(abs(phi_fem - phi_analytic)));
norm((phi_fem - phi_analytic))

%% ---------------- helper: linear shape functions ----------------------
function [N,dNdxi] = shapeLin(xi)
    N     = [1-xi; xi];
    dNdxi = [-1;    1];
end
% function [N,dNdxi] = shapeLin(xi)
%     N      = [2*(xi-0.5)*(xi-1);  4*xi*(1-xi); 2*xi*(xi-0.5)];
%     dNdxi  = [2*( (xi-0.5)+(xi-1) ); 4*(1 - 2*xi); 2*( (xi-0.5)+xi )];
% end

function [r_iso,drdxi,L] = isomap(re, N, dNdxi)
    drdxi = re.' * dNdxi;   % dr/dξ
    r_iso = re.' * N;       % physical r at GP
    L     = drdxi;          % Jacobian
end

