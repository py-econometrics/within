# Part 3: Local Solvers and Approximate Cholesky

This is Part 3 of the algorithm documentation for the `within` solver. It describes the local solve strategies used within each Schwarz subdomain: SDDM transformation, Schur complement reduction, and approximate Cholesky factorization.

**Series overview**:
- [Part 1: Fixed Effects and Block Iterative Methods](1_fixed_effects_and_block_methods.md)
- [Part 2: Preconditioned Krylov Solvers and Schwarz Decomposition](2_solver_architecture.md)
- **Part 3: Local Solvers and Approximate Cholesky** (this document)
- [Part 4: Implementation Notes](4_implementation.md)

**Prerequisites**: Part 1 (Gramian block structure), Part 2 (Schwarz framework, SDDM connection).

---

## Notation

Symbols from Parts [1](1_fixed_effects_and_block_methods.md) and [2](2_solver_architecture.md), plus:

| Symbol | Meaning |
|--------|---------|
| $n_q, n_r$ | Number of active levels in the local subdomain for factors $q, r$ |
| $C$ | Cross-tabulation block restricted to the subdomain (local) |
| $L$ | Sign-flipped SDDM matrix |
| $S$ | Schur complement of the eliminated block |
| $\tilde{L}$ | Approximate Cholesky factor |

---

## 1. The Local System

Each subdomain requires solving a system $A_i z_i = r_i$ where $A_i$ is the bipartite Gramian block for a factor pair restricted to a connected component. The local operator is:

$$
A_i = G_{\text{local}} = \begin{pmatrix} D_q & C \\ C^\top & D_r \end{pmatrix}
$$

where $D_q$, $D_r$ are the (diagonal) weighted-count matrices and $C$ is the cross-tabulation, all restricted to the component's active levels. This is an $(n_q + n_r) \times (n_q + n_r)$ sparse matrix.

Two solver strategies are available: full SDDM factorization and Schur complement reduction.

---

## 2. SDDM / Laplacian Connection

To apply the approximate Cholesky factorization (which requires a Laplacian or SDDM input), the bipartite Gramian is transformed via sign-flip (as described in [Part 2, Section 2.3](2_solver_architecture.md#23-sddm-connection-via-sign-flip)):

$$
L = \begin{pmatrix} D_q & -C \\ -C^\top & D_r \end{pmatrix}
$$

This is an SDDM matrix (and a graph Laplacian when row sums are exactly zero).

The local solve wrapper handles the sign convention:

1. **Before solve**: negate the second block of the RHS ($r[n_q \ldots] \leftarrow -r[n_q \ldots]$), subtract the mean
2. **Solve**: $L z = r$ via approximate Cholesky
3. **After solve**: negate the second block of the solution ($z[n_q \ldots] \leftarrow -z[n_q \ldots]$)

When the sign-flipped matrix is not exactly a Laplacian (rows don't sum to exactly zero due to numerical issues or structural reasons), **Gremban augmentation** adds one extra "ground" node connected to all others, absorbing the row-sum deficit. The augmented system is one dimension larger ($n_q + n_r + 1$) but is guaranteed to be a valid Laplacian.

The `LocalSolveStrategy` enum tracks which case applies:

| Variant | Sign-flip? | Augmented? | Solve dimension |
|---------|-----------|------------|-----------------|
| `Laplacian` | No | No | $n_q + n_r$ |
| `LaplacianGramian` | Yes | No | $n_q + n_r$ |
| `GramianAugmented` | Yes | Yes | $n_q + n_r + 1$ |
| `Sddm` | No | Yes | $n_q + n_r + 1$ |

---

## 3. Full SDDM Factorization

The simplest local solver constructs the full SDDM matrix $L$ (or its Gremban augmentation) and factors it with approximate Cholesky. The factor is stored and reused across Krylov iterations. This approach is straightforward but may be expensive for large components.

---

## 4. Schur Complement Reduction

For the bipartite structure, block Gaussian elimination can reduce the system size before factorization. Given:

$$
\begin{pmatrix} D_q & -C \\ -C^\top & D_r \end{pmatrix}
\begin{pmatrix} z_q \\ z_r \end{pmatrix}
= \begin{pmatrix} b_q \\ b_r \end{pmatrix}
$$

Eliminate the larger block (say $D_q$ when $m_q \geq m_r$). Since $D_q$ is diagonal, elimination is exact and cheap:

$$
S\, z_r = b_r + C^\top D_q^{-1} b_q
$$

where the **Schur complement** is:

$$
S = D_r - C^\top D_q^{-1} C
$$

The Schur complement $S$ is an $m_r \times m_r$ Laplacian on the smaller factor's levels, with edge weights that capture the indirect connections through the eliminated factor. After solving for $z_r$, back-substitution recovers:

$$
z_q = D_q^{-1}(b_q + C\, z_r)
$$

### 4.1 Exact Schur complement

For each row $i$ of the kept block, scatter into a dense workspace:

$$
S[i, j] = D_{\text{keep}}[i] \cdot \delta_{ij} - \sum_{k} \frac{C_{\text{keep} \to \text{elim}}[i,k] \cdot C_{\text{elim} \to \text{keep}}[k,j]}{D_{\text{elim}}[k]}
$$

The workspace is reset sparsely (only touched entries) after each row. Rows are computed in parallel.

### 4.2 Approximate Schur complement (clique-tree sampling)

The exact Schur complement $S = D_{\text{keep}} - C_{\text{k} \to \text{e}} \, D_{\text{elim}}^{-1} \, C_{\text{e} \to \text{k}}$ can be decomposed as a sum of rank-1 contributions, one per eliminated vertex:

$$
S = D_{\text{keep}} - \sum_{k=1}^{n_{\text{elim}}} \frac{1}{D_{\text{elim}}[k]} \, c_k \, c_k^\top
$$

where $c_k$ is the column of $C_{\text{k} \to \text{e}}$ corresponding to eliminated vertex $k$ (i.e., the edge weights from $k$ to its neighbors in the keep-block). Each rank-1 term $c_k c_k^\top / D_{\text{elim}}[k]$ is a weighted clique on the neighbors of $k$ — if vertex $k$ has $d$ neighbors, this clique has $\binom{d}{2}$ edges, which can be expensive to materialize.

**The key insight** from Gao, Kyng, and Spielman (2025): each rank-1 clique can be approximated by a random spanning tree of the star graph centered at $k$. A spanning tree has only $d - 1$ edges (linear in degree, not quadratic), and its edge weights can be chosen so that the expected Laplacian of the tree equals the rank-1 clique's Laplacian. This makes clique-tree sampling an **unbiased estimator** of each eliminated vertex's Schur complement contribution.

Concretely, for eliminated vertex $k$ with neighbors $u_1, \ldots, u_d$ having weights $w_1, \ldots, w_d$ and $s_k = D_{\text{elim}}[k]$:
- The exact clique adds edge $(u_i, u_j)$ with weight $w_i w_j / s_k$
- The sampled tree adds $d - 1$ edges whose expected Laplacian matches the clique Laplacian

Edges from all eliminated vertices are sorted, deduplicated (summing weights for duplicate edges), and merged across threads via a parallel reduce tree. The result is assembled into a symmetric CSR Laplacian.

### 4.3 Dense fast-path

When the reduced system is small ($\min(m_q, m_r) \leq 24$ by default), the Schur complement is computed directly as a dense matrix and factored with exact dense Cholesky. The factorization operates on the *anchored* $(m_{\text{keep}} - 1) \times (m_{\text{keep}} - 1)$ principal minor (dropping one coordinate that is fixed to zero, since the Laplacian is rank-deficient). Forward and backward triangular solves use direct indexing for performance.

---

## 5. Approximate Cholesky Factorization

The approximate Cholesky algorithm (from the external `approx-chol` crate) factors a Laplacian $L$ into an approximate lower-triangular factor $\tilde{L}$ such that $\tilde{L}\tilde{L}^\top \approx L$. It is used both for full SDDM factorization (Section 3) and for the sparse Schur complement (Section 4.2).

The algorithm is a modified Gaussian elimination that processes vertices in a random order. At each step, eliminating a vertex produces fill — but instead of materializing exact fill, it samples an approximation with far fewer edges.

### 5.1 Elimination as rank-1 update

Eliminating vertex $v$ with edge weights $w_1, \ldots, w_d$ to neighbors $u_1, \ldots, u_d$ and diagonal $s_v = \sum_i w_i$ produces the rank-1 Schur complement contribution:

$$
\Delta S = \frac{1}{s_v} \begin{pmatrix} w_1 \\ \vdots \\ w_d \end{pmatrix} \begin{pmatrix} w_1 & \cdots & w_d \end{pmatrix}
$$

As a Laplacian, this is a complete graph (clique) on $N(v)$ with edge $(u_i, u_j)$ weighted $w_i w_j / s_v$.

Standard Cholesky would materialize all $\binom{d}{2}$ clique edges as fill, potentially leading to dense factors.

### 5.2 Clique-tree sampling

Instead of materializing the $O(d^2)$ clique edges, sample a random spanning tree of the star graph $\{v\} \cup N(v)$. The tree has exactly $d - 1$ edges. Edge weights are set so that the expected Laplacian of the tree equals the clique Laplacian — an **unbiased estimator** with $O(d)$ fill instead of $O(d^2)$.

This is exactly the same idea used in Section 4.2 for the approximate Schur complement: the Schur complement reduction eliminates the entire larger block at once (each vertex independently), while the approximate Cholesky eliminates vertices one by one during factorization. The core operation — replace a clique with a sampled spanning tree — is identical.

### 5.3 AC(k) variant

The `split` parameter controls how many independent spanning trees are sampled per star:

- **AC(1)** (`split = 1`): one tree per elimination step. Maximum sparsity, highest variance.
- **AC(k)** (`split = k`): $k$ independent trees sampled, edge weights averaged. Higher $k$ reduces variance (better approximation quality) at the cost of up to $k(d-1)$ fill edges per elimination step.

### 5.4 Triangular solve

The approximate factor $\tilde{L}$ is stored in sparse format. Solving $\tilde{L}\tilde{L}^\top z = b$ proceeds via forward substitution ($\tilde{L} y = b$) then backward substitution ($\tilde{L}^\top z = y$).

### 5.5 Properties

The key property is that $\mathbb{E}[\tilde{L}\tilde{L}^\top] = L$ (unbiased), and the factorization runs in $O(m \log m)$ expected time for bounded-degree graphs. For the fixed-effects Gramian, the local systems are typically sparse enough that this is effectively linear.

**Reference**: Gao, Kyng, and Spielman (2025) provide the full algorithm and analysis for approximate Cholesky factorization via clique-tree sampling.

---

## References

**Gao, Y., Kyng, R., & Spielman, D. A.** (2025). *Robust and Practical Solution of Laplacian Equations by Approximate Gaussian Elimination*. arXiv:2303.00709. Primary reference for the approximate Cholesky factorization via clique-tree sampling (AC(k) algorithm), Schur complement approximation, and Gremban augmentation.

**Gremban, K. D.** (1996). *Combinatorial Preconditioners for Sparse, Symmetric, Diagonally Dominant Linear Systems*. PhD thesis, Carnegie Mellon University. SDDM-to-Laplacian augmentation technique.
