# Part 2: Preconditioned Krylov Solvers and Schwarz Decomposition

This is Part 2 of the algorithm documentation for the `within` solver. It describes the three-layer solver architecture, the graph structure that drives the decomposition, the Krylov outer iteration, and the Schwarz preconditioner framework.

**Series overview**:
- [Part 1: Fixed Effects and Block Iterative Methods](1_fixed_effects_and_block_methods.md)
- **Part 2: Preconditioned Krylov Solvers and Schwarz Decomposition** (this document)
- [Part 3: Local Solvers and Approximate Cholesky](3_local_solvers.md)

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

3. **Local solvers** — each subdomain system is a bipartite Gramian block that becomes a graph Laplacian after sign-flip, factored in nearly-linear time using approximate Cholesky (see [Part 3](3_local_solvers.md)).

### Why this combination

The Gramian $G$ for fixed-effects models is typically large and sparse, with block structure governed by the factor-pair interactions. Three properties motivate the design:

- **Block-bipartite structure**: each factor pair $(q,r)$ induces a bipartite subgraph. Connected components of these subgraphs form natural subdomains with limited overlap.

- **Laplacian connection**: after a sign-flip transformation (Section 2), each bipartite block becomes a graph Laplacian, enabling the use of nearly-linear-time Laplacian solvers.

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

| Full interaction graph | Worker–Firm subgraph |
|:---:|:---:|
| ![Full graph](images/graph_plain.svg) | ![Worker–Firm bipartite subgraph](images/graph_bipartite_wf.svg) |

Continuing the Worker/Firm/Year example from [Part 1](1_fixed_effects_and_block_methods.md): extracting just the Worker–Firm edges (right) from the full graph (left) gives the bipartite graph of $C_{WF}$ with edges W1–F1, W1–F2, W2–F1, W3–F2. Running DFS from any node reaches all others (W1 bridges both firms), so there is a single connected component containing all 5 DOFs {W1, W2, W3, F1, F2}. The same holds for the Worker–Year and Firm–Year pairs, giving 3 subdomains with one component each.

Without W1's mobility (if W1 only worked at F1), the Worker–Firm graph would split into two components: {W1, W2, F1} and {W3, F2}, yielding two independent subdomains for this pair.

### 2.3 Laplacian connection via sign-flip

The bipartite block $G_{qr}$ has non-negative off-diagonal entries (cross-tabulation counts), so it is not directly a graph Laplacian. However, negating the off-diagonal blocks produces one:

$$
L_{qr} = \begin{pmatrix} D_q & -C_{qr} \\ -C_{qr}^\top & D_r \end{pmatrix}
$$

This is a graph Laplacian: symmetric, non-positive off-diagonal entries, and row sums exactly zero. The zero row-sum property holds because every observation at level $j$ of factor $q$ has exactly one level in factor $r$, so $D_q[j,j] = \sum_k C_{qr}[j,k]$ (and symmetrically for the $r$-rows). This is true regardless of the number of factors $Q$ — the diagonal $D_q[j,j]$ counts total observation weight at level $j$, and every such observation contributes to exactly one entry of $C_{qr}$.

The transform is an involution: solving $L_{qr} z = b$ and flipping the sign of the second block of $z$ recovers the solution to $G_{qr} z' = b'$.

This Laplacian structure is exploited by the local solvers ([Part 3](3_local_solvers.md)), which convert bipartite Gramian blocks to Laplacian form and apply approximate Cholesky factorization — a nearly-linear-time algorithm designed for Laplacian systems.

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

### 3.3 Convergence tolerance vs. demean quality

The Krylov convergence criterion $\|r_k\|_2 / \|b\|_2 \leq \text{tol}$ controls the normal-equation residual, not the demeaning error directly. These are related but distinct quantities. The normal equations $G\alpha = D^\top W y$ are a means to an end — the goal is the demeaned outcome $\tilde{y} = y - D\alpha$, and its quality depends on how errors in $\alpha$ propagate through $D$.

An error $\delta\alpha$ in the coefficient vector produces a demeaning error $D\delta\alpha$. The ratio $\|D\delta\alpha\| / \|\delta\alpha\|$ depends on the spectrum of $D^\top D$ (i.e., the observation counts per level), so a normal-equation tolerance of $10^{-8}$ does not guarantee $10^{-8}$ accuracy in $\tilde{y}$ — it can be better or worse depending on the data geometry. In practice, the normal-equation residual is a reliable proxy: when it is small, the demeaned values are accurate enough for downstream inference.

**Iterative refinement.** When higher accuracy is needed (or when approximate local solvers introduce additional error), the solution can be improved by iterative refinement: solve $G\alpha_0 = b$ to moderate tolerance, compute the residual $r = b - G\alpha_0$, solve $G\delta = r$, and update $\alpha \leftarrow \alpha_0 + \delta$. Each refinement step reuses the existing preconditioner and reduces the error by roughly the same factor as the original solve. This is particularly relevant when the approximate Cholesky local solvers ([Part 3](3_local_solvers.md)) limit preconditioner quality — refinement recovers accuracy without rebuilding the factorization. The solver reports the independently verified residual $\|G\alpha - b\| / \|b\|$ after each solve regardless of convergence status.

---

## 4. Schwarz Domain Decomposition

[Part 1, Section 5](1_fixed_effects_and_block_methods.md#5-the-domain-decomposition-perspective) introduced the Schwarz perspective and contrasted factor-level with factor-pair decompositions. This section provides the full algorithmic details.

### 4.1 Space decomposition framework

The Schwarz method decomposes the global DOF space $\mathbb{R}^m$ into overlapping subspaces $S_i \subset \mathbb{R}^m$, each defined by a set of global DOF indices. Let $R_i: \mathbb{R}^m \to S_i$ be the restriction operator that extracts the subdomain DOFs, and $R_i^\top: S_i \to \mathbb{R}^m$ the prolongation that scatters them back.

The local operator on subdomain $i$ is $A_i = R_i G R_i^\top$, the principal submatrix of $G$ restricted to subdomain $i$'s DOFs. Solving $A_i z_i = r_i$ is the local solve.

**Reference**: Xu (1992) provides the abstract space decomposition framework underlying Schwarz methods.

### 4.2 Partition of unity

When subdomains overlap (a DOF belongs to multiple subdomains), corrections must be weighted to avoid double-counting. The two-sided additive Schwarz formula applies weights on both the restriction and prolongation sides (Section 4.3), so the partition-of-unity must satisfy:

$$
\sum_{i} R_i^\top \tilde{D}_i^2 R_i = I
$$

For a DOF $j$ appearing in $c_j$ subdomains, each weight is set to:

$$
\tilde{d}_{i,j} = \frac{1}{\sqrt{c_j}}, \qquad c_j = |\{k : j \in V_k\}|
$$

so that $c_j \times (1/\sqrt{c_j})^2 = 1$ at every DOF. This ensures that the preconditioner correctly partitions the global correction without under- or over-correcting at overlap regions.

When a subdomain has no overlapping DOFs (all counts are 1, so all weights are 1), the weights are stored as a single integer count rather than a per-DOF vector, avoiding unnecessary allocation.

### 4.3 Additive Schwarz

The additive Schwarz preconditioner applies all local solves independently and sums the weighted corrections:

$$
M^{-1}_{\text{add}} r = \sum_{i=1}^{N_s} R_i^\top \tilde{D}_i A_i^+ \tilde{D}_i R_i r
$$

For each subdomain $i$:
1. **Restrict**: $r_i = \tilde{D}_i R_i r$ (gather DOFs from global residual, apply PoU weights)
2. **Local solve**: $z_i = A_i^+ r_i$
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
&\quad z_i = A_i^+ r_i \\
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
5. **Compute partition-of-unity weights**: count how many subdomains each DOF belongs to, assign $w = 1/\sqrt{\text{count}}$.

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

3. Compute partition-of-unity weights (1/sqrt(count) per DOF)

4. For each subdomain in parallel:
   a. Build local Laplacian (sign-flip) or compute Schur complement
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

**Xu, J.** (1992). *Iterative Methods by Space Decomposition and Subspace Correction*. SIAM Review, 34(4), 581–613. Provides the abstract space decomposition framework for additive and multiplicative Schwarz methods.

**Toselli, A. & Widlund, O. B.** (2005). *Domain Decomposition Methods — Algorithms and Theory*. Springer. Comprehensive reference for the theory and convergence analysis of Schwarz domain decomposition methods.

