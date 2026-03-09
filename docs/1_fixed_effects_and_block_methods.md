# Part 1: Fixed Effects and Block Iterative Methods

This is Part 1 of the algorithm documentation for the `within` solver. It introduces the fixed-effects estimation problem, derives the normal equations and their block structure, describes the classical demeaning algorithm and its limitations, and motivates the domain decomposition approach in [Part 2](2_solver_architecture.md).

**Series overview**:
- **Part 1: Fixed Effects and Block Iterative Methods** (this document)
- [Part 2: Preconditioned Krylov Solvers and Schwarz Decomposition](2_solver_architecture.md)
- [Part 3: Local Solvers and Approximate Cholesky](3_local_solvers.md)

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

The Gramian $G$ is symmetric positive semi-definite and always singular: within each connected component of the factor interaction graph, a constant can be shifted between factors without changing $D\alpha$. The system is always consistent, and the solver (starting from zero) converges to the minimum-norm solution.

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

Consider an AKM-style panel with $n = 6$ observations and $Q = 3$ factors (Worker, Firm, Year). Worker W1 moves from Firm F1 to F2 — this mobility is what connects the two firms in the estimation graph.

| Obs | Worker ($f_1$) | Firm ($f_2$) | Year ($f_3$) | Weight | $y$ |
|-----|---------|------|------|--------|------|
| 1 | W1 | F1 | Y1 | 1 | 3.2 |
| 2 | W1 | F2 | Y2 | 1 | 4.1 |
| 3 | W2 | F1 | Y1 | 1 | 2.8 |
| 4 | W2 | F1 | Y2 | 1 | 3.9 |
| 5 | W3 | F2 | Y1 | 1 | 5.0 |
| 6 | W3 | F2 | Y2 | 1 | 4.5 |

Factor 1 (Worker) has $m_1 = 3$ levels: {W1, W2, W3}. Factor 2 (Firm) has $m_2 = 2$ levels: {F1, F2}. Factor 3 (Year) has $m_3 = 2$ levels: {Y1, Y2}. Total DOFs: $m = 7$.

The Gramian has $Q = 3$ diagonal blocks and $\binom{3}{2} = 3$ cross-tabulation blocks:

$$
G = \begin{pmatrix}
{\color{royalblue}D_W} & {\color{gray}C_{WF}} & {\color{gray}C_{WY}} \\
{\color{gray}C_{WF}^\top} & {\color{crimson}D_F} & {\color{gray}C_{FY}} \\
{\color{gray}C_{WY}^\top} & {\color{gray}C_{FY}^\top} & {\color{forestgreen}D_Y}
\end{pmatrix}
= \left(\begin{array}{ccc|cc|cc}
{\color{royalblue}2} & {\color{royalblue}0} & {\color{royalblue}0} & {\color{gray}1} & {\color{gray}1} & {\color{gray}1} & {\color{gray}1} \\
{\color{royalblue}0} & {\color{royalblue}2} & {\color{royalblue}0} & {\color{gray}2} & {\color{gray}0} & {\color{gray}1} & {\color{gray}1} \\
{\color{royalblue}0} & {\color{royalblue}0} & {\color{royalblue}2} & {\color{gray}0} & {\color{gray}2} & {\color{gray}1} & {\color{gray}1} \\
\hline
{\color{gray}1} & {\color{gray}2} & {\color{gray}0} & {\color{crimson}3} & {\color{crimson}0} & {\color{gray}2} & {\color{gray}1} \\
{\color{gray}1} & {\color{gray}0} & {\color{gray}2} & {\color{crimson}0} & {\color{crimson}3} & {\color{gray}1} & {\color{gray}2} \\
\hline
{\color{gray}1} & {\color{gray}1} & {\color{gray}1} & {\color{gray}2} & {\color{gray}1} & {\color{forestgreen}3} & {\color{forestgreen}0} \\
{\color{gray}1} & {\color{gray}1} & {\color{gray}1} & {\color{gray}1} & {\color{gray}2} & {\color{forestgreen}0} & {\color{forestgreen}3}
\end{array}\right)
$$

**Reading the blocks.** $D_W$ is $3 \times 3$ (one row/column per worker) with 2s on the diagonal because each worker appears in exactly 2 observations (e.g. W1 in obs 1, 2). Off-diagonals are zero because no observation belongs to two workers. $D_F$ is $2 \times 2$ with 3s on the diagonal because each firm appears in 3 observations (F1 in obs 1, 3, 4; F2 in obs 2, 5, 6). The cross-tabulation block $C_{WY}$ is $3 \times 2$ (workers $\times$ years); entry $[j,k]$ counts observations where worker $j$ is observed in year $k$. Here every worker appears once per year, so $C_{WY}$ is all ones.

The Gramian's sparsity pattern defines an interaction graph on all $m = 7$ DOFs. Each node is a factor level, each edge a nonzero cross-tabulation entry $C_{qr}[j,k]$:

![Gramian interaction graph](images/graph_plain.svg)

Workers (pink), Firms (blue), Years (yellow). Each edge is a nonzero cross-tabulation entry; no edges connect DOFs within the same factor. The graph is connected because W1's mobility between F1 and F2 bridges the two firms; without it, the graph would split into disconnected components (independent sub-problems).

### 3.3 Key properties

Two properties of $G$ that drive the algorithmic design:

1. **The diagonal blocks $D_q$ are diagonal matrices** — trivially invertible. This makes the classical demeaning algorithm (Section 4) cheap per iteration.

2. **The cross-tabulation blocks $C_{qr}$ are typically sparse** — an entry is nonzero only when at least one observation has that specific (level-of-$q$, level-of-$r$) combination.

---

## 4. Classical Solution: Iterative Demeaning

### 4.1 The demeaning algorithm

The classical method sweeps through factors one at a time, updating each factor's coefficients while holding the others fixed. For factor $q$, each coefficient is set to the weighted average of the partial residual at its level:

$$
\alpha_{q,j} \leftarrow \frac{\sum_{i:\, f_q(i)=j} w_i\, (y_i - \sum_{r \neq q} \alpha_{r,f_r(i)})}{\sum_{i:\, f_q(i)=j} w_i}
$$

This is **demeaning by factor $q$**: compute the residual $y - \text{(all other factor effects)}$, then replace each coefficient with the weighted group mean of that residual.

One full sweep updates all $Q$ factors in order. The algorithm is equivalent to **block Gauss-Seidel** on the normal equations — each factor's update solves the diagonal block $D_q$ against the current residual restricted to that factor's DOFs:

$$
D_q\,\alpha_q^{(k+1)} = b_q - \sum_{r < q} C_{qr}\,\alpha_r^{(k+1)} - \sum_{r > q} C_{qr}\,\alpha_r^{(k)}
$$

Factors $r < q$ have already been updated in this sweep; factors $r > q$ have not.

| Viewpoint | Space | One factor-$q$ step | Key object |
|---|---|---|---|
| Demeaning | $\mathbb{R}^n$ (observations) | weighted mean of partial residual | raw data $(y, w)$ |
| Block Gauss-Seidel | $\mathbb{R}^m$ (coefficients) | $D_q^{-1}(b_q - \sum C_{qr}\alpha_r)$ | Gramian blocks $(D_q, C_{qr})$ |

These are two notations for the same computation.

### 4.2 The running example

Continuing the Worker/Firm/Year example, one full sweep processes $Q = 3$ factors in order:

**Step 1** — Update Worker, holding Firm and Year fixed:
Each worker's coefficient becomes the weighted average of $(y_i - \text{firm effect} - \text{year effect})$ for observations at that worker.

**Step 2** — Update Firm, using **updated** Worker values but **stale** Year values:
Each firm's coefficient becomes the weighted average of $(y_i - \text{worker effect} - \text{year effect})$, but the year effects are still from the previous iteration.

**Step 3** — Update Year, using updated Worker and updated Firm values:
Only the last factor in the sweep sees fully updated values from all other factors.

Only the last factor in the sweep sees fully updated values from all other factors. Workers were updated with stale firm and year effects; firms were updated with stale year effects. With $Q = 3$, two out of three updates use partially stale information — illustrating the degradation described in Section 4.4.

Repeat until convergence.

### 4.3 Convergence rate

The convergence rate is governed by how "entangled" the factor subspaces are. For two factors, the error contracts by $\cos^2(\theta_F)$ per sweep, where $\theta_F$ is the angle between the factor subspaces:

![Convergence: high vs low mobility](images/convergence_zigzag.svg)

- **High mobility** (workers move between many firms): the factor subspaces are nearly orthogonal, $\cos(\theta_F)$ is small, convergence is fast.
- **Low mobility** (workers stuck at one firm): the factor subspaces are nearly collinear, $\cos(\theta_F) \to 1$, convergence degrades sharply — the algorithm zigzags with little progress.

For $Q > 2$, any near-collinear pair bottlenecks the entire iteration.


### 4.4 Limitations

Three structural limitations of iterative demeaning:

1. **No way to improve the local solve** — each block solve is already exact ($D_q$ is diagonal), so there is no knob to turn within a subdomain. The only option is to iterate more.

2. **Cross-factor structure is ignored** — the local solve for factor $q$ knows nothing about the coupling $C_{qr}$. When convergence is slow, the coupling is strong — yet the algorithm cannot exploit the cross-tabulation structure that causes the difficulty.

3. **Degradation with more factors** — for $Q > 2$, each factor's update uses stale values from factors not yet processed in the current sweep. The effective convergence rate worsens as the number of interacting pairs grows.

---

## 5. The Domain Decomposition Perspective

### 5.1 Demeaning as a domain decomposition method

Iterative demeaning partitions the coefficient vector $\alpha \in \mathbb{R}^m$ into $Q$ blocks — one per factor — and sweeps through them sequentially. Each step solves a small system (the diagonal block $D_q$) using the current residual restricted to that block's DOFs.

In domain decomposition terminology, this is a **multiplicative Schwarz** method with $Q$ non-overlapping subdomains, one per factor. The decomposition has cheap local solves but captures none of the cross-factor coupling:

![Factor-level decomposition](images/graph_factor_level.svg)

The same interaction graph from Section 3.2, now grouped into factor-level subdomains. Every edge crosses a subdomain boundary — the local operators $D_q$ are trivially diagonal, with no internal edges. The local solves are trivial but they capture **none** of the cross-factor coupling.

### 5.2 The key idea: factor-pair subdomains

`within` uses a fundamentally different decomposition: subdomains are **factor pairs**, not individual factors.

For each pair $(q, r)$, the subdomain contains all DOFs from both factors. The local operator is the full bipartite block:

$$
A_{qr} = \begin{pmatrix} D_q & C_{qr} \\ C_{qr}^\top & D_r \end{pmatrix}
$$

This captures the complete interaction between the two factors — the diagonal counts **and** the cross-tabulation. The price: these bipartite systems are too large to solve exactly in practice, so the local solvers use **approximate** factorizations (Schur complement reduction + approximate Cholesky, see [Part 3](3_local_solvers.md)). This is the central trade-off in `within`:

The difference in what each local solve "sees" is dramatic:

![Factor-level vs factor-pair local solve](images/local_solve_comparison.svg)

| Property | Factor-level (demeaning) | Factor-pair (`within`) |
|---|---|---|
| Local solve | **Exact** (diagonal inversion) | **Approximate** (sampled Cholesky) |
| Coupling captured | None — ignores $C_{qr}$ entirely | Full pairwise interaction |
| Number of subdomains | $Q$ | $\leq \binom{Q}{2} \times$ (components) |
| Overlap | None | Yes — each DOF appears in $Q - 1$ pairs |

Demeaning solves each factor *exactly* but learns nothing about cross-factor structure. Factor-pair subdomains solve each pair *approximately* but capture the coupling that makes convergence slow in the first place. The net win: the approximate pair-solves carry so much more information per iteration that far fewer outer iterations are needed — even though each individual local solve is not exact.

### 5.3 Continuing the example

The Worker/Firm/Year example ($Q = 3$) produces $\binom{3}{2} = 3$ factor-pair subdomains:

| Subdomain | Factor pair | DOFs | Size |
|-----------|-------------|------|------|
| 1 | (Worker, Firm) | W1, W2, W3, F1, F2 | 5 |
| 2 | (Worker, Year) | W1, W2, W3, Y1, Y2 | 5 |
| 3 | (Firm, Year) | F1, F2, Y1, Y2 | 4 |

Each DOF appears in $Q - 1 = 2$ subdomains, requiring partition-of-unity weights to prevent double-counting (see [Part 2, Section 4.2](2_solver_architecture.md#42-partition-of-unity)).

The same interaction graph, now with three overlapping factor-pair subdomains drawn around it:

![Factor-pair decomposition](images/graph_factor_pair.svg)

The Worker–Firm subdomain (red) covers the top two rows. The Firm–Year subdomain (blue) covers the bottom two. The Worker–Year subdomain (green) wraps around the Firms in a C-shape. Every edge is now *inside* a subdomain. Each node sits in exactly 2 of the 3 boxes, which is why partition-of-unity weights are needed.

### 5.4 Where this leads

The factor-pair decomposition raises three algorithmic questions:

1. **How to solve the local systems efficiently?** → Approximate Cholesky factorization with Schur complement reduction ([Part 3](3_local_solvers.md))
2. **How to combine the local corrections?** → Additive or multiplicative Schwarz with partition-of-unity weights ([Part 2](2_solver_architecture.md))
3. **How to drive the global iteration?** → Preconditioned CG or GMRES ([Part 2](2_solver_architecture.md))

---

## References

**Correia, S.** (2016). *A Feasible Estimator for Linear Models with Multi-Way Fixed Effects*. Working paper. Describes the fixed-effects normal equations, their block structure, and iterative solution via alternating projections.

**Xu, J.** (1992). *Iterative Methods by Space Decomposition and Subspace Correction*. SIAM Review, 34(4), 581–613. Provides the abstract space decomposition framework for additive and multiplicative Schwarz methods.

**Toselli, A. & Widlund, O. B.** (2005). *Domain Decomposition Methods — Algorithms and Theory*. Springer. Comprehensive reference for the theory and convergence analysis of Schwarz methods.
