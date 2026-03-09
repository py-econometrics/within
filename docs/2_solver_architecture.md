# Part 2: Preconditioned Krylov Solvers and Schwarz Decomposition

This is Part 2 of the algorithm documentation for the `within` solver. It describes the three-layer solver architecture, the graph structure that drives the decomposition, the Krylov outer iteration, and the Schwarz preconditioner framework.

**Series overview**:
- [Part 1: Fixed Effects and Block Iterative Methods](1_fixed_effects_and_block_methods.md)
- **Part 2: Preconditioned Krylov Solvers and Schwarz Decomposition** (this document)
- [Part 3: Local Solvers and Approximate Cholesky](3_local_solvers.md)
- [Part 4: Implementation Notes](4_implementation.md)

**Prerequisites**: Part 1 (problem formulation, Gramian block structure, Block Gauss-Seidel as multiplicative Schwarz).

---

## Notation

Symbols from [Part 1](1_fixed_effects_and_block_methods.md), plus:

| Symbol | Meaning |
|--------|---------|
| $R_i$ | Restriction operator for subdomain $i$ |
| $\tilde{D}_i$ | Partition-of-unity weight matrix for subdomain $i$ |
| $A_i$ | Local operator $R_i G R_i^\top$ |
| $M^{-1}$ | Preconditioner (approximate inverse of $G$) |
| $N_s$ | Number of subdomains |
| $\text{tol}$ | Convergence tolerance (default $10^{-8}$) |

---

## 1. Three-Layer Architecture

The solver combines three algorithmic ideas in a layered architecture:

```
┌─────────────────────────────────────────┐
│  Krylov solver (CG or GMRES)            │  Outer iteration
│    solves G α = D^T W y                 │
├─────────────────────────────────────────┤
│  Schwarz preconditioner                 │  Applied once per
│    M⁻¹ = Σ Rᵢᵀ D̃ᵢ Aᵢ⁻¹ D̃ᵢ Rᵢ          │  Krylov iteration
├─────────────────────────────────────────┤
│  Local solvers (approx Cholesky)        │  One per subdomain,
│    solve Aᵢ zᵢ = rᵢ                    │  independent & parallel
└─────────────────────────────────────────┘
```

1. **Krylov solver** — conjugate gradient (CG) or restarted GMRES — iterates toward the solution, using the preconditioner to accelerate convergence.

2. **Schwarz preconditioner** — decomposes the global system into overlapping subdomains derived from the Gramian's block structure, applies local solves independently (additive) or sequentially (multiplicative), and combines corrections using partition-of-unity weights.

3. **Local solvers** — each subdomain system is a small SDDM (symmetric diagonally dominant) matrix that can be factored in nearly-linear time using approximate Cholesky factorization (see [Part 3](3_local_solvers.md)).

### Why this combination

The Gramian $G$ for fixed-effects models is typically large and sparse, with block structure governed by the factor-pair interactions. Three properties motivate the design:

- **Block-bipartite structure**: each factor pair $(q,r)$ induces a bipartite subgraph. Connected components of these subgraphs form natural subdomains with limited overlap.

- **SDDM / Laplacian connection**: after a sign-flip transformation (Section 2), each bipartite block becomes an SDDM matrix (and a graph Laplacian when row sums vanish), enabling the use of nearly-linear-time Laplacian solvers via Gremban augmentation when needed.

- **Spectral acceleration**: Schwarz preconditioning clusters the eigenvalues of $M^{-1}G$, reducing the Krylov iteration count from $O(\sqrt{\kappa})$ (unpreconditioned CG) to a count that depends on the quality of the local solves rather than the global condition number.

---

## 2. Graph Structure of the Gramian

Part 1 derived the block structure $G = \mathcal{D} + \mathcal{C}$ with diagonal blocks $D_q$ and cross-tabulation blocks $C_{qr}$. This section describes the graph-theoretic properties that drive the domain decomposition.

### 2.1 Factor-pair bipartite blocks

Each cross-tabulation block $C_{qr}$ defines a weighted bipartite graph: the left vertices are the $m_q$ levels of factor $q$, the right vertices are the $m_r$ levels of factor $r$, and the edge weight between $j$ and $k$ is $C_{qr}[j,k]$ (nonzero when at least one observation has $f_q = j$ and $f_r = k$).

The full factor-pair block is:

$$
G_{qr} = \begin{pmatrix} D_q & C_{qr} \\ C_{qr}^\top & D_r \end{pmatrix}
$$

where $D_q$ and $D_r$ are the diagonal blocks restricted to the active levels in this pair.

### 2.2 Connected components as subdomains

The bipartite graph of $C_{qr}$ may have multiple connected components. Each connected component defines an independent subproblem: the rows and columns of $G_{qr}$ can be permuted so that disconnected components are block-diagonal. These components become the subdomains of the Schwarz preconditioner.

Continuing the example from [Part 1](1_fixed_effects_and_block_methods.md): the bipartite graph of $C_{12}$ has edges A–2020, A–2021, B–2020, B–2021, C–2021, C–2022. Running DFS from any node reaches all others, so there is a single connected component. The entire 6-DOF system forms one subdomain.

If firms A and B only traded in 2020 and firm C only traded in 2022, the graph would have two components: {A, B, 2020} and {C, 2022}, yielding two independent subdomains.

### 2.3 SDDM connection via sign-flip

The bipartite block $G_{qr}$ has non-negative off-diagonal entries (cross-tabulation counts), so it is not directly a graph Laplacian. However, negating the off-diagonal blocks produces an SDDM matrix:

$$
L_{qr} = \begin{pmatrix} D_q & -C_{qr} \\ -C_{qr}^\top & D_r \end{pmatrix}
$$

This is a symmetric diagonally dominant (SDDM) matrix: symmetric, non-positive off-diagonal entries, and non-negative diagonal entries that satisfy $D_q[j,j] \geq \sum_k C_{qr}[j,k]$ (with equality when all observations at level $j$ of factor $q$ have an active partner in factor $r$). When the row sums are exactly zero, $L_{qr}$ is a graph Laplacian; in general it is SDDM. The transform is an involution: solving $L_{qr} z = b$ and flipping the sign of the second block of $z$ recovers the solution to $G_{qr} z' = b'$.

When $L_{qr}$ is not exactly a Laplacian (nonzero row sums), **Gremban augmentation** adds a single "ground" node connected to every other node, absorbing the row-sum deficit. The augmented system is one dimension larger but is a proper Laplacian, compatible with the approximate Cholesky solver.

This connection is exploited by the local solvers ([Part 3](3_local_solvers.md)), which convert bipartite Gramian blocks to SDDM form and apply approximate Cholesky factorization — a nearly-linear-time algorithm designed for Laplacian and SDDM systems.

**Reference**: Correia (2016) describes the fixed-effects normal equations and their block structure. Gremban (1996) introduces the augmentation technique for SDDM-to-Laplacian reduction.

---

## 3. Preconditioned Krylov Methods

The outer solver is a Krylov method that iteratively solves $G\alpha = b$ where $b = D^\top W y$.

### 3.1 Left-preconditioned Conjugate Gradient

When the preconditioner $M^{-1}$ is symmetric positive-definite (as with additive Schwarz), CG is used. The left-preconditioned CG algorithm:

$$
\boxed{
\begin{aligned}
&\textbf{Input: } G,\; M^{-1},\; b,\; \text{tol},\; \text{maxiter} \\
&x_0 = 0, \quad r_0 = b, \quad z_0 = M^{-1} r_0, \quad p_0 = z_0 \\
&\rho_0 = r_0^\top z_0, \quad \|b\| = \|r_0\|_2 \\[4pt]
&\textbf{for } k = 0, 1, 2, \dots \\
&\quad q_k = G p_k \\
&\quad \gamma_k = p_k^\top q_k \\
&\quad \textbf{if } \gamma_k \leq 0: \text{ exit (indefinite)} \\
&\quad \alpha_k = \rho_k / \gamma_k \\
&\quad x_{k+1} = x_k + \alpha_k p_k \\
&\quad r_{k+1} = r_k - \alpha_k q_k \\
&\quad \textbf{if } \|r_{k+1}\|_2 / \|b\| \leq \text{tol}: \text{ converged} \\
&\quad z_{k+1} = M^{-1} r_{k+1} \\
&\quad \rho_{k+1} = r_{k+1}^\top z_{k+1} \\
&\quad \beta_k = \rho_{k+1} / \rho_k \\
&\quad p_{k+1} = z_{k+1} + \beta_k p_k
\end{aligned}
}
$$

**Convergence criterion**: $\|r_k\|_2 / \|b\|_2 \leq \text{tol}$, checked after each residual update. The algorithm also exits early on indefinite curvature ($p^\top G p \leq 0$) and stagnation ($|\rho_{k+1}| < \varepsilon \cdot |\rho_0|$).

### 3.2 Right-preconditioned GMRES(m)

When the preconditioner is non-symmetric (as with multiplicative Schwarz), GMRES with restarts is used. The right-preconditioned formulation solves $G M^{-1} (M\alpha) = b$, applying the preconditioner to the search direction rather than the residual.

The algorithm maintains an orthonormal Krylov basis $V = [v_1, \dots, v_k]$ via Arnoldi iteration with Modified Gram-Schmidt:

$$
\boxed{
\begin{aligned}
&\textbf{Input: } G,\; M^{-1},\; b,\; \text{tol},\; m \text{ (restart)},\; \text{maxiter} \\
&x_0 = 0, \quad \beta = \|b\|_2, \quad v_1 = b/\beta \\[4pt]
&\textbf{repeat } \text{(restart cycles):} \\
&\quad g = (\beta, 0, \dots, 0)^\top \in \mathbb{R}^{m+1} \\
&\quad \textbf{for } j = 1, \dots, m: \\
&\quad\quad z_j = M^{-1} v_j \\
&\quad\quad w = G z_j \\
&\quad\quad \textbf{for } i = 1, \dots, j: \quad h_{ij} = w^\top v_i, \quad w \leftarrow w - h_{ij} v_i \\
&\quad\quad h_{j+1,j} = \|w\|_2 \\
&\quad\quad \text{Apply Givens rotations to } h_{\cdot,j} \text{ and } g \\
&\quad\quad \textbf{if } |g_{j+1}| \leq \text{tol} \cdot \|b\|: \text{ converged} \\
&\quad\quad v_{j+1} = w / h_{j+1,j} \\
&\quad \text{Solve upper-triangular } R y = g_{1:j} \\
&\quad x \leftarrow x + Z_j y \quad \text{where } Z_j = [z_1, \dots, z_j]
\end{aligned}
}
$$

The Givens rotations maintain an implicit QR factorization of the upper Hessenberg matrix, and $|g_{j+1}|$ tracks the residual norm without explicit computation. Lucky breakdown ($h_{j+1,j} \approx 0$) indicates the Krylov subspace is exhausted.

### 3.3 Implicit vs. explicit operator

The Gramian-vector product $G x = D^\top W (D x)$ can be computed in two ways:

- **Implicit**: compute $t = Dx$ (gather), then $D^\top W t$ (weighted scatter). No matrix is ever formed; cost is $O(nQ)$ per product.

- **Explicit**: build $G$ once as a CSR sparse matrix. Cost is $O(\text{nnz}(G))$ per product, with an upfront $O(nQ)$ assembly cost.

The implicit operator avoids the memory and construction cost of the explicit CSR matrix, which can be significant when $m$ is large. The explicit operator is required when multiplicative Schwarz is used with the sparse Gramian residual updater, which scatters Gramian rows directly into the residual after each subdomain correction.

### 3.4 Post-solve residual

After convergence, the solver independently verifies the residual in the normal-equation space:

$$
\text{final\_residual} = \frac{\|G\alpha - D^\top W y\|_2}{\max(\|D^\top W y\|_2,\; 10^{-15})}
$$

This is reported regardless of whether the Krylov method converged.

---

## 4. Schwarz Domain Decomposition

[Part 1, Section 5](1_fixed_effects_and_block_methods.md#5-the-domain-decomposition-perspective) introduced the Schwarz perspective and contrasted factor-level with factor-pair decompositions. This section provides the full algorithmic details.

### 4.1 Space decomposition framework

The Schwarz method decomposes the global DOF space $V = \mathbb{R}^m$ into overlapping subspaces $V_i \subset V$, each defined by a set of global DOF indices. Let $R_i: V \to V_i$ be the restriction operator that extracts the subdomain DOFs, and $R_i^\top: V_i \to V$ the prolongation that scatters them back.

The local operator on subdomain $i$ is $A_i = R_i G R_i^\top$, the principal submatrix of $G$ restricted to subdomain $i$'s DOFs. Solving $A_i z_i = r_i$ is the local solve.

**Reference**: Xu (1992) provides the abstract space decomposition framework underlying Schwarz methods.

### 4.2 Partition of unity

When subdomains overlap (a DOF belongs to multiple subdomains), corrections must be weighted to avoid double-counting. The partition-of-unity assigns each DOF $j$ in subdomain $i$ a weight:

$$
\tilde{d}_{i,j} = \frac{1}{|\{k : j \in V_k\}|}
$$

where the denominator counts how many subdomains contain DOF $j$. By construction:

$$
\sum_{i} R_i^\top \tilde{D}_i^2 R_i = I
$$

where $\tilde{D}_i = \text{diag}(\tilde{d}_{i,1}, \dots, \tilde{d}_{i,|V_i|})$. This identity ensures that the preconditioner correctly partitions the global correction.

When a subdomain has no overlapping DOFs (all counts are 1), the weights are stored as a single integer count rather than a per-DOF vector, avoiding unnecessary allocation.

### 4.3 Additive Schwarz

The additive Schwarz preconditioner applies all local solves independently and sums the weighted corrections:

$$
M^{-1}_{\text{add}} r = \sum_{i=1}^{N_s} R_i^\top \tilde{D}_i A_i^{-1} \tilde{D}_i R_i r
$$

For each subdomain $i$:
1. **Restrict**: $r_i = \tilde{D}_i R_i r$ (gather DOFs from global residual, apply PoU weights)
2. **Local solve**: $z_i = A_i^{-1} r_i$
3. **Prolongate**: scatter $\tilde{D}_i z_i$ back to global, accumulating across subdomains

All subdomains are processed in parallel. The additive preconditioner is symmetric, making it compatible with CG.

### 4.4 Multiplicative Schwarz

The multiplicative Schwarz preconditioner processes subdomains sequentially, updating the residual after each correction:

$$
\boxed{
\begin{aligned}
&z = 0, \quad r_{\text{work}} = r \\
&\textbf{for } i = 1, \dots, N_s: \\
&\quad r_i = \tilde{D}_i R_i\, r_{\text{work}} \\
&\quad z_i = A_i^{-1} r_i \\
&\quad z \leftarrow z + R_i^\top \tilde{D}_i z_i \\
&\quad r_{\text{work}} \leftarrow r_{\text{work}} - G R_i^\top \tilde{D}_i z_i
\end{aligned}
}
$$

The sequential residual update $r_{\text{work}} \leftarrow r - G(R_i^\top \tilde{D}_i z_i)$ allows each subdomain to "see" corrections from earlier subdomains, generally improving convergence versus additive Schwarz. However, the preconditioner is non-symmetric, requiring GMRES instead of CG. An optional backward sweep (reverse subdomain order) can be appended for a symmetric variant.

### 4.5 Subdomain construction

Subdomains are derived from the factor-pair structure of the Gramian:

1. **Enumerate factor pairs**: all $\binom{Q}{2}$ unordered pairs $(q, r)$ with $q < r$.
2. **Build cross-tabulation**: for each pair, scan observations to build the sparse bipartite block $C_{qr}$ and diagonal vectors $D_q$, $D_r$.
3. **Find connected components**: run DFS on the bipartite graph of $C_{qr}$ to identify independent components.
4. **Create subdomains**: each component becomes a subdomain. Its global DOF indices are the active levels of factors $q$ and $r$ within that component.
5. **Compute partition-of-unity weights**: count how many subdomains each DOF belongs to, assign $w = 1/\text{count}$.

Factor pairs are processed in parallel. When there is only one connected component per pair (the common case), the full cross-tabulation is shared via reference counting rather than copied.

---

## 5. Full Algorithm Summary

### 5.1 Setup phase

```
Input: categories[n][Q], y[n], weights[n], solver_params, precond_config

1. Build WeightedDesign from categories and weights
   - Infer n_levels per factor by scanning observations
   - Compute global offsets: offset[q] = sum of n_levels[0..q]

2. For each factor pair (q, r) in parallel:
   a. Build cross-tabulation C_qr, diagonal blocks D_q, D_r
   b. Find connected components of the bipartite graph
   c. For each component: create subdomain with global DOF indices

3. Compute partition-of-unity weights (1/count per DOF)

4. For each subdomain in parallel:
   a. Build local SDDM or compute Schur complement
   b. Factor with approximate Cholesky (or dense Cholesky for small systems)

5. Assemble Schwarz preconditioner from subdomain entries

6. Optionally: build explicit Gramian CSR from the same cross-tabulation blocks
```

### 5.2 Solve phase

```
Input: WeightedDesign, preconditioner M⁻¹, right-hand side y

1. Project to normal equations: b = Dᵀ W y

2. Dispatch Krylov solver (configured independently):
   - CG is used with additive Schwarz (symmetric preconditioner)
   - GMRES is used with multiplicative Schwarz (non-symmetric)

3. Solve G α = b with preconditioner M⁻¹
   - Each iteration: one G·x product + one M⁻¹·r application
   - Converge when ‖rₖ‖ / ‖b‖ ≤ tol

4. Compute demeaned residuals: e = y - D α

5. Verify final residual: ‖G α - b‖ / ‖b‖
```

### 5.3 Complexity

| Phase | Cost | Notes |
|-------|------|-------|
| Cross-tabulation build | $O(n Q^2)$ | One observation scan per factor pair |
| Connected components | $O(m + \text{nnz})$ per pair | DFS on bipartite graph |
| Schur complement (exact) | $O(m_{\text{keep}} \cdot \text{nnz}_{\text{local}})$ | Row-workspace, parallelized |
| Schur complement (approx) | $O(\text{nnz}_{\text{local}})$ | One tree per star |
| Approximate Cholesky | $O(m_{\text{local}} \log m_{\text{local}})$ | Per subdomain |
| Gramian-vector product (implicit) | $O(nQ)$ | Per Krylov iteration |
| Gramian-vector product (explicit) | $O(\text{nnz}(G))$ | Per Krylov iteration |
| Preconditioner application | $\sum_i O(m_i)$ | Local triangular solves |

The total cost per Krylov iteration is dominated by the operator application and preconditioner application. The setup cost is amortized if the preconditioner is reused across multiple right-hand sides (batch solve).

---

## References

**Correia, S.** (2016). *A Feasible Estimator for Linear Models with Multi-Way Fixed Effects*. Working paper. Describes the fixed-effects normal equations and their block structure.

**Gremban, K. D.** (1996). *Combinatorial Preconditioners for Sparse, Symmetric, Diagonally Dominant Linear Systems*. PhD thesis, Carnegie Mellon University. Introduces the augmentation technique for SDDM-to-Laplacian reduction.

**Xu, J.** (1992). *Iterative Methods by Space Decomposition and Subspace Correction*. SIAM Review, 34(4), 581–613. Provides the abstract space decomposition framework for additive and multiplicative Schwarz methods.

**Toselli, A. & Widlund, O. B.** (2005). *Domain Decomposition Methods — Algorithms and Theory*. Springer. Comprehensive reference for the theory and convergence analysis of Schwarz domain decomposition methods.
