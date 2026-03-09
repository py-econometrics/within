# Part 1: Fixed Effects and Block Iterative Methods

This is Part 1 of the algorithm documentation for the `within` solver. It introduces the fixed-effects estimation problem, derives the normal equations and their block structure, describes the classical alternating-projection algorithm, and reinterprets it through the lens of domain decomposition — motivating the more sophisticated solver described in [Part 2](2_solver_architecture.md).

**Series overview**:
- **Part 1: Fixed Effects and Block Iterative Methods** (this document)
- [Part 2: Preconditioned Krylov Solvers and Schwarz Decomposition](2_solver_architecture.md)
- [Part 3: Local Solvers and Approximate Cholesky](3_local_solvers.md)
- [Part 4: Implementation Notes](4_implementation.md)

---

## Notation

| Symbol | Meaning |
|--------|---------|
| $n$ | Number of observations |
| $Q$ | Number of factors (fixed-effect grouping variables) |
| $m_q$ | Number of levels (categories) in factor $q$ |
| $m = \sum_q m_q$ | Total degrees of freedom |
| $D$ | Design matrix ($n \times m$), with exactly $Q$ ones per row |
| $W$ | Diagonal weight matrix ($n \times n$), $W = \text{diag}(w_1, \dots, w_n)$ |
| $G$ | Gramian matrix, $G = D^\top W D$ ($m \times m$) |
| $y$ | Response vector ($n \times 1$) |
| $\alpha$ | Fixed-effects coefficient vector ($m \times 1$) |
| $D_q$ | Diagonal block of $G$ for factor $q$ (weighted level counts) |
| $C_{qr}$ | Cross-tabulation block of $G$ for factor pair $(q, r)$ |

---

## 1. The Fixed-Effects Model

A panel dataset records $n$ observations, each classified by $Q$ categorical factors. Factor $q$ has $m_q$ distinct levels. The linear fixed-effects model is:

$$
y_i = \sum_{q=1}^{Q} \alpha_{q, f_q(i)} + \varepsilon_i, \qquad i = 1, \dots, n
$$

where $f_q(i) \in \{1, \dots, m_q\}$ is the level of factor $q$ for observation $i$ and $\alpha_{q,j}$ is the coefficient for level $j$ of factor $q$.

In matrix form: $y = D\alpha + \varepsilon$, where $D$ is the $n \times m$ design matrix. Each row of $D$ has exactly $Q$ ones, one in each factor's column block.

---

## 2. Weighted Normal Equations

The weighted least-squares estimate satisfies the normal equations:

$$
G \alpha = D^\top W y, \qquad G = D^\top W D
$$

where $W = \text{diag}(w_1, \dots, w_n)$ is a diagonal weight matrix. The unweighted case corresponds to $W = I$.

The Gramian $G$ is symmetric positive semi-definite. It is always singular: within each connected component of the factor interaction graph, a constant can be shifted between factors without changing $D\alpha$, giving at least one null-space direction per component. The system is always consistent, and the solver (starting from zero) converges to a minimum-norm solution.

---

## 3. Block Structure of the Gramian

### 3.1 Factor-block partition

The $m \times m$ Gramian inherits a natural block structure from the factor partition. With columns ordered by factor ($m_1$ columns for factor 1, then $m_2$ for factor 2, etc.):

$$
G = \begin{pmatrix}
D_1 & C_{12} & C_{13} & \cdots & C_{1Q} \\
C_{12}^\top & D_2 & C_{23} & \cdots & C_{2Q} \\
C_{13}^\top & C_{23}^\top & D_3 & \cdots & C_{3Q} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
C_{1Q}^\top & C_{2Q}^\top & C_{3Q}^\top & \cdots & D_Q
\end{pmatrix}
$$

The blocks are:

- **Diagonal blocks** $D_q$ ($m_q \times m_q$, diagonal): $D_q[j,j] = \sum_{i:\, f_q(i) = j} w_i$ — the weighted count of observations at level $j$ of factor $q$.

- **Cross-tabulation blocks** $C_{qr}$ ($m_q \times m_r$, sparse): $C_{qr}[j,k] = \sum_{i:\, f_q(i) = j,\; f_r(i) = k} w_i$ — the weighted count of observations simultaneously at level $j$ of factor $q$ and level $k$ of factor $r$.

There are $Q$ diagonal blocks and $\binom{Q}{2}$ cross-tabulation blocks.

### 3.2 A concrete example

Consider a dataset with $n = 6$ observations and $Q = 2$ factors (Firm and Year):

| Obs | Firm ($f_1$) | Year ($f_2$) | Weight | $y$ |
|-----|-------|------|--------|------|
| 1 | A | 2020 | 1 | 3.2 |
| 2 | A | 2021 | 1 | 4.1 |
| 3 | B | 2020 | 1 | 2.8 |
| 4 | B | 2021 | 1 | 3.9 |
| 5 | C | 2021 | 1 | 5.0 |
| 6 | C | 2022 | 1 | 4.5 |

Factor 1 (Firm) has $m_1 = 3$ levels: {A, B, C}. Factor 2 (Year) has $m_2 = 3$ levels: {2020, 2021, 2022}. Total DOFs: $m = 6$.

The Gramian is:

$$
G = \begin{pmatrix}
D_1 & C_{12} \\
C_{12}^\top & D_2
\end{pmatrix}
= \left(\begin{array}{ccc|ccc}
2 & 0 & 0 & 1 & 1 & 0 \\
0 & 2 & 0 & 1 & 1 & 0 \\
0 & 0 & 2 & 0 & 1 & 1 \\
\hline
1 & 1 & 0 & 2 & 0 & 0 \\
1 & 1 & 1 & 0 & 3 & 0 \\
0 & 0 & 1 & 0 & 0 & 1
\end{array}\right)
$$

The diagonal blocks record how many observations each level has. The cross-tabulation $C_{12}[\text{A}, 2020] = 1$ records that firm A appears once in year 2020.

### 3.3 Key properties

Two properties of $G$ that drive the algorithmic design:

1. **The diagonal blocks $D_q$ are diagonal matrices** — trivially invertible. This makes Block Gauss-Seidel (Section 4) cheap per iteration.

2. **The cross-tabulation blocks $C_{qr}$ are sparse** — an entry is nonzero only when at least one observation has that specific (level-of-$q$, level-of-$r$) combination. The sparsity pattern of $C_{qr}$ encodes which factor levels interact.

---

## 4. Classical Solution: Alternating Projection

### 4.1 The algorithm

The classical approach to solving $G\alpha = b$ (where $b = D^\top W y$) partitions the unknowns by factor and sweeps through one factor at a time. For each factor $q$, update its coefficients by solving the $q$-th block row of the normal equations while holding all other factors fixed:

$$
\alpha_q^{(k+1)} = D_q^{-1} \left( b_q - \sum_{r \neq q} C_{qr} \alpha_r^{(\cdot)} \right)
$$

where $\alpha_r^{(\cdot)}$ uses the most recent value — updated values for factors already processed in this sweep, and previous-iteration values for factors not yet processed.

Since $D_q$ is diagonal, each component reduces to a weighted average:

$$
\alpha_{q,j}^{(k+1)} = \frac{1}{D_q[j,j]} \left( b_{q,j} - \sum_{r \neq q} \sum_{k} C_{qr}[j,k] \, \alpha_{r,k}^{(\cdot)} \right)
$$

There is an equivalent **observation-space** view: compute the partial residual $e_i = y_i - \sum_{r \neq q} \alpha_{r, f_r(i)}$ (everything except factor $q$), then set $\alpha_{q,j}$ to the weighted mean of $e_i$ over observations at level $j$ of factor $q$. This is "demeaning by factor" — projecting the residual onto the column space of factor $q$.

### 4.2 The running example

Continuing the Firm/Year example, one full sweep looks like:

**Step 1** — Update Firm ($q = 1$), holding Year fixed at $\alpha_2^{(k)}$:

$$
\alpha_{\text{Firm A}}^{(k+1)} = \frac{1}{2}\bigl(b_{\text{A}} - 1 \cdot \alpha_{2020}^{(k)} - 1 \cdot \alpha_{2021}^{(k)}\bigr)
$$

Each firm's coefficient becomes the weighted average of $(y_i - \text{year effect})$ for observations at that firm.

**Step 2** — Update Year ($q = 2$), using the **updated** Firm values $\alpha_1^{(k+1)}$:

$$
\alpha_{2021}^{(k+1)} = \frac{1}{3}\bigl(b_{2021} - 1 \cdot \alpha_{\text{A}}^{(k+1)} - 1 \cdot \alpha_{\text{B}}^{(k+1)} - 1 \cdot \alpha_{\text{C}}^{(k+1)}\bigr)
$$

Each year's coefficient becomes the weighted average of $(y_i - \text{firm effect})$ for observations in that year.

Repeat until convergence.

### 4.3 Equivalence to Block Gauss-Seidel

Write the normal equations in block form:

$$
G\alpha = (\mathcal{D} + \mathcal{C})\alpha = b
$$

where $\mathcal{D} = \text{blkdiag}(D_1, \dots, D_Q)$ is the block diagonal and $\mathcal{C} = G - \mathcal{D}$ contains all cross-tabulation blocks. **Block Gauss-Seidel** on this splitting is:

$$
\mathcal{D} \,\alpha^{(k+1)} = b - \mathcal{C} \,\alpha^{(\cdot)}
$$

which is exactly the alternating-projection iteration: factor-by-factor update with $D_q^{-1}$ as the trivial "block solve." The use of most-recent values (Gauss-Seidel style, not Jacobi) makes each factor's update see corrections from factors already processed in the current sweep.

This is the algorithm used by `reghdfe` (Correia, 2016), typically combined with conjugate-gradient-like acceleration.

### 4.4 Convergence and limitations

Block Gauss-Seidel on a symmetric positive semi-definite system always converges (for consistent right-hand sides). The error satisfies:

$$
\|\alpha^{(k)} - \alpha^*\| \leq C \cdot \rho^k
$$

where $\rho$ is the spectral radius of the Gauss-Seidel iteration matrix $(\mathcal{D}+\mathcal{L})^{-1}\mathcal{U}$ (with $\mathcal{L}$, $\mathcal{U}$ the strict lower/upper block-triangular parts of $\mathcal{C}$). This is bounded above by the Jacobi spectral radius $\rho(\mathcal{D}^{-1}\mathcal{C})$, which can be close to 1 for datasets with strongly interacting factors — meaning slow convergence. Three limitations stand out:

1. **No way to improve the local solve** — each block solve is already exact ($D_q$ is diagonal), so the only knob is the iteration strategy.

2. **Cross-factor structure is ignored** — the local solve for factor $q$ knows nothing about the coupling $C_{qr}$ with other factors. This information enters only through the residual update.

3. **Degradation with more factors** — for $Q > 2$, each factor's correction is contaminated by stale values from factors not yet updated in the current sweep.

---

## 5. The Domain Decomposition Perspective

The alternating-projection algorithm is a special case of the **Schwarz domain decomposition** framework. Recognizing this opens the door to more powerful decomposition strategies.

### 5.1 Schwarz methods in brief

A Schwarz method decomposes the global DOF space $V = \mathbb{R}^m$ into overlapping subspaces $V_i \subset V$. For each subspace, let $R_i: V \to V_i$ be the restriction operator (extracts the relevant DOFs) and $A_i = R_i G R_i^\top$ the local operator. The method applies local corrections:

- **Multiplicative Schwarz**: process subdomains sequentially, updating the global residual after each:
$$
z \leftarrow z + R_i^\top A_i^{-1} R_i r_{\text{current}}
$$

- **Additive Schwarz**: process all subdomains in parallel and sum the weighted corrections:
$$
M^{-1} r = \sum_i R_i^\top \tilde{D}_i A_i^{-1} \tilde{D}_i R_i\, r
$$
where $\tilde{D}_i$ are partition-of-unity weights that prevent double-counting when subdomains overlap.

### 5.2 Block Gauss-Seidel as multiplicative Schwarz

Alternating projection is **multiplicative Schwarz with factor-level subdomains**:

| Schwarz concept | Block GS instantiation |
|---|---|
| Number of subdomains | $Q$ (one per factor) |
| Subdomain $q$ DOFs | $\{m_1 + \cdots + m_{q-1} + 1, \;\dots,\; m_1 + \cdots + m_q\}$ |
| Overlap | **None** — factor subdomains are disjoint |
| Local operator $A_q$ | $D_q$ (diagonal) |
| Local solve cost | $O(m_q)$ — just division |
| Partition-of-unity weights | All 1 (no overlap) |

The decomposition has $Q$ small, cheap subdomains that cover all DOFs exactly once. The local solves are trivial — but they capture **none** of the cross-factor coupling.

### 5.3 Factor-pair decomposition

`within` uses a fundamentally different decomposition: subdomains are **factor pairs**, not individual factors.

For each pair $(q, r)$ with $q < r$, the subdomain contains all DOFs from both factors $q$ and $r$ that appear in the same connected component of the bipartite graph defined by $C_{qr}$. The local operator is the full bipartite block:

$$
A_{qr} = \begin{pmatrix} D_q & C_{qr} \\ C_{qr}^\top & D_r \end{pmatrix}
$$

This captures the complete interaction between the two factors — the diagonal counts **and** the cross-tabulation.

| Property | Factor-level (Block GS) | Factor-pair (`within`) |
|---|---|---|
| Number of subdomains | $Q$ | $\leq \binom{Q}{2} \times$ (components) |
| Overlap | None | Yes — each DOF appears in $Q - 1$ pairs |
| Local operator | $D_q$ (diagonal, trivial) | $G_{qr}$ (bipartite, SDDM-class) |
| Cross-factor coupling | Ignored | Fully captured per pair |
| Local solve cost | $O(m_q)$ | $O(m_q + m_r)$ to $O((m_q + m_r) \log(m_q + m_r))$ |

The local systems are now proper sparse linear systems — no longer trivially diagonal — but they belong to the SDDM (symmetric diagonally dominant M-matrix) class, for which nearly-linear-time solvers exist. The trade-off: harder local solves in exchange for dramatically fewer outer iterations, because the preconditioner captures far more of the global Gramian's structure.

### 5.4 Continuing the example

In the Firm/Year example, the factor-pair decomposition produces a single subdomain (since the bipartite graph has one connected component) containing all 6 DOFs:

$$
A_{12} = \begin{pmatrix} D_{\text{Firm}} & C_{12} \\ C_{12}^\top & D_{\text{Year}} \end{pmatrix} = G
$$

The local solve captures the **entire** system. With a good approximate factorization, the preconditioner is nearly an exact inverse — convergence in 1–2 iterations rather than the many sweeps required by alternating projection.

For more factors ($Q = 3$: Firm, Year, Industry), there would be 3 factor-pair subdomains: (Firm, Year), (Firm, Industry), (Year, Industry). Each DOF appears in 2 subdomains, requiring partition-of-unity weights of $\frac{1}{2}$.

### 5.5 Where this leads

The factor-pair decomposition raises three algorithmic questions:

1. **How to solve the local SDDM systems efficiently?** → Approximate Cholesky factorization with Schur complement reduction ([Part 3](3_local_solvers.md))
2. **How to combine the local corrections?** → Additive or multiplicative Schwarz with partition-of-unity weights ([Part 2](2_solver_architecture.md))
3. **How to drive the global iteration?** → Preconditioned CG or GMRES ([Part 2](2_solver_architecture.md))

---

## References

**Correia, S.** (2016). *A Feasible Estimator for Linear Models with Multi-Way Fixed Effects*. Working paper. Describes the fixed-effects normal equations, their block structure, and iterative solution via alternating projections.

**Xu, J.** (1992). *Iterative Methods by Space Decomposition and Subspace Correction*. SIAM Review, 34(4), 581–613. Provides the abstract space decomposition framework for additive and multiplicative Schwarz methods.

**Toselli, A. & Widlund, O. B.** (2005). *Domain Decomposition Methods — Algorithms and Theory*. Springer. Comprehensive reference for the theory and convergence analysis of Schwarz methods.
