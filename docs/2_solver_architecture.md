# Part 2: Preconditioned Krylov Solvers and Schwarz Decomposition

This is Part 2 of the algorithm documentation for the `within` solver. It describes the three-layer solver architecture, the graph structure that drives the decomposition, the Krylov outer iteration, and the Schwarz preconditioner framework.

**Series overview**:
- [Part 1: Fixed Effects and Block Iterative Methods](1_fixed_effects_and_block_methods.md)
- **Part 2: Preconditioned Krylov Solvers and Schwarz Decomposition** (this document)
- [Part 3: Local Solvers and Approximate Cholesky](3_local_solvers.md)

**Prerequisites**: Part 1 (problem formulation, Gramian block structure, demeaning as multiplicative Schwarz).

---

## 1. Three-Layer Architecture

The solver combines three algorithmic ideas in a layered architecture:

![Three-layer solver architecture](images/three_layer_architecture.svg)

1. **Krylov solver** — conjugate gradient (CG) or restarted GMRES — iterates toward the solution, using the preconditioner to accelerate convergence.

2. **Schwarz preconditioner** — decomposes the global system into overlapping subdomains derived from the Gramian's block structure, applies local solves independently (additive) or sequentially (multiplicative), and combines corrections using partition-of-unity weights.

3. **Local solvers** — each subdomain system is a bipartite Gramian block that becomes a graph Laplacian after sign-flip, factored in nearly-linear time using approximate Cholesky (see [Part 3](3_local_solvers.md)).

### Why this combination

As discussed in [Part 1, Section 5.2](1_fixed_effects_and_block_methods.md#52-the-key-idea-factor-pair-subdomains), the central trade-off is **exact solves on single factors** (demeaning) vs. **approximate solves on factor pairs** (`within`). The three layers exist to make this trade-off pay off:

- **Block-bipartite structure**: each factor pair $(q,r)$ induces a bipartite subgraph. Connected components of these subgraphs form natural subdomains with limited overlap.

- **Laplacian connection**: after a sign-flip transformation (Section 2), each bipartite block becomes a graph Laplacian. This unlocks nearly-linear-time *approximate* solvers — exact factorization would be too expensive for large subdomains, but the approximate Cholesky factorization ([Part 3](3_local_solvers.md)) produces factors that are accurate enough to make the Krylov solver converge in very few iterations.

- **Spectral acceleration**: the Krylov outer solver compensates for the approximate nature of the local solves. Even though each preconditioner application is inexact, the Krylov iteration refines the solution globally. The preconditioner clusters the eigenvalues of $M^{-1}G$, reducing the iteration count from $O(\sqrt{\kappa})$ (unpreconditioned demeaning) to a count determined by the quality of the local solves rather than the global condition number.

---

## 2. Graph Structure of the Gramian

Part 1 derived the block structure $G = \mathcal{D} + \mathcal{C}$ with diagonal blocks $D_q$ and cross-tabulation blocks $C_{qr}$. This section describes the graph-theoretic properties that drive the domain decomposition.

### 2.1 Factor-pair bipartite blocks

Each cross-tabulation block $C_{qr}$ defines a weighted bipartite graph: the left vertices are the $m_q$ levels of factor $q$, the right vertices are the $m_r$ levels of factor $r$, and the edge weight between $j$ and $k$ is $C_{qr}[j,k]$ (nonzero when at least one observation has $f_q = j$ and $f_r = k$).

The full factor-pair block is:

$$
G_{qr} = \begin{pmatrix} D_q & C_{qr} \\ C_{qr}^\top & D_r \end{pmatrix}
$$

### 2.2 Connected components as subdomains

The bipartite graph of $C_{qr}$ may have multiple connected components. Each connected component defines an independent subproblem and becomes a subdomain of the Schwarz preconditioner.

| Full interaction graph | Worker–Firm subgraph |
|:---:|:---:|
| ![Full graph](images/graph_plain.svg) | ![Worker–Firm bipartite subgraph](images/graph_bipartite_wf.svg) |

Continuing the Worker/Firm/Year example from [Part 1](1_fixed_effects_and_block_methods.md): extracting just the Worker–Firm edges (right) gives the bipartite graph of $C_{WF}$. Running DFS from any node reaches all others (W1 bridges both firms), so there is a single connected component containing all 5 DOFs. Without W1's mobility, the graph would split into two components: {W1, W2, F1} and {W3, F2}, yielding two independent subdomains.

### 2.3 Laplacian connection via sign-flip

The bipartite block $G_{qr}$ has non-negative off-diagonal entries, so it is not directly a graph Laplacian. Negating the off-diagonal blocks produces one:

$$
L_{qr} = \begin{pmatrix} D_q & -C_{qr} \\ -C_{qr}^\top & D_r \end{pmatrix}
$$

This is a valid graph Laplacian: symmetric, non-positive off-diagonal entries, and zero row sums. The zero row-sum property holds because every observation at level $j$ of factor $q$ has exactly one level in factor $r$, so $D_q[j,j] = \sum_k C_{qr}[j,k]$.

The transform is an involution: solving $L_{qr} z = b$ and flipping the sign of one block recovers the solution to the original system. This Laplacian structure is exploited by the local solvers ([Part 3](3_local_solvers.md)).

---

## 3. Krylov Outer Iteration

The outer solver is a Krylov method that iteratively solves $G\alpha = b$ where $b = D^\top W y$. At each step, the Krylov solver computes a residual $r = b - G\alpha$, applies the preconditioner to get a correction direction $z = M^{-1}r$, and updates the solution.

The key idea: instead of solving $G\alpha = b$ directly (which may take thousands of demeaning iterations), we use the Schwarz preconditioner $M^{-1}$ to "pre-digest" the residual at each step, pointing the solver toward the answer much faster.

![Preconditioned vs unpreconditioned convergence](images/preconditioned_convergence.svg)

### 3.1 Solver selection

- **CG** (conjugate gradient) is used when the preconditioner is symmetric — this is the case with additive Schwarz. CG is optimal for symmetric positive-definite systems.

- **GMRES** (generalized minimal residual) is used when the preconditioner is non-symmetric — this is the case with multiplicative Schwarz, where the sequential processing breaks symmetry.

### 3.2 Convergence criterion

The solver converges when $\|r_k\|_2 / \|b\|_2 \leq \text{tol}$ (default $10^{-8}$). This controls the normal-equation residual, not the demeaning error directly, but in practice it is a reliable proxy: when the normal-equation residual is small, the demeaned values are accurate enough for downstream inference.

The solver reports the independently verified residual $\|G\alpha - b\| / \|b\|$ after each solve.

**Iterative refinement.** When higher accuracy is needed, the solution can be improved by solving $G\delta = (b - G\alpha_0)$ and updating $\alpha \leftarrow \alpha_0 + \delta$. Each refinement step reuses the existing preconditioner and reduces the error by roughly the same factor as the original solve.

---

## 4. Schwarz Domain Decomposition

[Part 1, Section 5](1_fixed_effects_and_block_methods.md#5-the-domain-decomposition-perspective) introduced the Schwarz perspective and contrasted factor-level with factor-pair decompositions. This section provides the full algorithmic details.

### 4.1 How it works

The Schwarz preconditioner decomposes the global system into overlapping subdomains — one per factor pair — and applies local solves to each. The local operator on subdomain $i$ is $A_i = R_i G R_i^\top$, the principal submatrix of $G$ restricted to that subdomain's DOFs.

Two variants exist — additive and multiplicative — differing in how the local corrections are combined:

![Additive vs multiplicative Schwarz](images/additive_vs_multiplicative.svg)

### 4.2 Partition of unity

When subdomains overlap (a DOF belongs to multiple subdomains), corrections must be weighted to avoid double-counting:

![Partition of unity](images/partition_of_unity.svg)

Each DOF $j$ that appears in $c_j$ subdomains gets weight $1/\sqrt{c_j}$ in each subdomain. The weights are applied on both the restriction and prolongation sides, so they contribute $c_j \times (1/\sqrt{c_j})^2 = 1$ — correctly partitioning the correction. In the running example, every DOF appears in exactly 2 subdomains, so every weight is $1/\sqrt{2}$.

### 4.3 Additive Schwarz

The additive Schwarz preconditioner applies all local solves independently and sums the weighted corrections:

$$
M^{-1}_{\text{add}} r = \sum_{i=1}^{N_s} R_i^\top \tilde{D}_i A_i^+ \tilde{D}_i R_i r
$$

For each subdomain: gather DOFs from the global residual with weights → local solve → scatter back with weights. All subdomains are processed in parallel. The preconditioner is symmetric, making it compatible with CG.

### 4.4 Multiplicative Schwarz

The multiplicative variant processes subdomains sequentially, updating the residual after each correction. Each subdomain "sees" corrections from earlier subdomains, generally improving convergence versus additive Schwarz. However, the sequential processing makes the preconditioner non-symmetric, requiring GMRES.

### 4.5 Subdomain construction

Subdomains are derived from the factor-pair structure of the Gramian:

1. **Enumerate factor pairs**: all $\binom{Q}{2}$ unordered pairs $(q, r)$.
2. **Build cross-tabulation**: for each pair, scan observations to build the sparse bipartite block $C_{qr}$ and diagonal vectors $D_q$, $D_r$.
3. **Find connected components**: run DFS on the bipartite graph of $C_{qr}$ to identify independent components.
4. **Create subdomains**: each component becomes a subdomain with its global DOF indices.
5. **Compute partition-of-unity weights**: count how many subdomains each DOF belongs to, assign $w = 1/\sqrt{\text{count}}$.

Factor pairs are processed in parallel.

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

2. Dispatch Krylov solver:
   - CG with additive Schwarz (symmetric preconditioner)
   - GMRES with multiplicative Schwarz (non-symmetric)

3. Solve G α = b with preconditioner M⁻¹
   - Each iteration: one G·x product + one M⁻¹·r application
   - Converge when ‖rₖ‖ / ‖b‖ ≤ tol

4. Compute demeaned residuals: e = y - D α

5. Verify final residual: ‖G α - b‖ / ‖b‖
```

---

## References

**Correia, S.** (2016). *A Feasible Estimator for Linear Models with Multi-Way Fixed Effects*. Working paper. Describes the fixed-effects normal equations and their block structure.

**Xu, J.** (1992). *Iterative Methods by Space Decomposition and Subspace Correction*. SIAM Review, 34(4), 581–613. Provides the abstract space decomposition framework for additive and multiplicative Schwarz methods.

**Toselli, A. & Widlund, O. B.** (2005). *Domain Decomposition Methods — Algorithms and Theory*. Springer. Comprehensive reference for the theory and convergence analysis of Schwarz domain decomposition methods.
