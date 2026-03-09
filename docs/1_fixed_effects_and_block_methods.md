# Part 1: Fixed Effects and Block Iterative Methods

This is Part 1 of the algorithm documentation for the `within` solver. It introduces the fixed-effects estimation problem, derives the normal equations and their block structure, describes the classical alternating-projection algorithm, and reinterprets it through the lens of domain decomposition — motivating the more sophisticated solver described in [Part 2](2_solver_architecture.md).

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

The Gramian $G$ is symmetric positive semi-definite. It is always singular: within each connected component of the factor interaction graph, a constant can be shifted between factors without changing $D\alpha$, giving at least one null-space direction per component. The system is always consistent ($b = D^\top W y$ lies in the column space of $G$), and the solver (starting from zero) converges to the minimum-norm solution $\alpha^* = G^+ b$, where $G^+$ denotes the Moore–Penrose pseudo-inverse.

The local operators $A_i$ arising in the domain decomposition (Section 5) are also singular — each connected bipartite block has a one-dimensional null space. We write $A_i^+$ for the pseudo-inverse throughout; all right-hand sides encountered are consistent, so $A_i^+ r$ is the unique minimum-norm solution.

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
D_W & C_{WF} & C_{WY} \\
C_{WF}^\top & D_F & C_{FY} \\
C_{WY}^\top & C_{FY}^\top & D_Y
\end{pmatrix}
= \left(\begin{array}{ccc|cc|cc}
2 & 0 & 0 & 1 & 1 & 1 & 1 \\
0 & 2 & 0 & 2 & 0 & 1 & 1 \\
0 & 0 & 2 & 0 & 2 & 1 & 1 \\
\hline
1 & 2 & 0 & 3 & 0 & 2 & 1 \\
1 & 0 & 2 & 0 & 3 & 1 & 2 \\
\hline
1 & 1 & 1 & 2 & 1 & 3 & 0 \\
1 & 1 & 1 & 1 & 2 & 0 & 3
\end{array}\right)
$$

The three block types are visible: $D_W = \text{diag}(2,2,2)$ counts observations per worker, $C_{WF}[\text{W1}, \text{F1}] = 1$ records that W1 appears at F1 once, and $C_{FY}[\text{F1}, \text{Y1}] = 2$ records that F1 has two observations in Y1.

The Gramian's sparsity pattern defines an interaction graph on all $m = 7$ DOFs. Each node is a factor level, each edge a nonzero cross-tabulation entry $C_{qr}[j,k]$:

![Gramian interaction graph](images/graph_plain.svg)

Workers (pink), Firms (blue), Years (yellow). Each edge is a nonzero cross-tabulation entry; no edges connect DOFs within the same factor. The graph is connected because W1's mobility between F1 and F2 bridges the two firms; without it, the graph would split into disconnected components (independent sub-problems).

### 3.3 Key properties

Two properties of $G$ that drive the algorithmic design:

1. **The diagonal blocks $D_q$ are diagonal matrices** — trivially invertible. This makes Block Gauss-Seidel (Section 4) cheap per iteration.

2. **The cross-tabulation blocks $C_{qr}$ are typically low-rank or sparse** — an entry is nonzero only when at least one observation has that specific (level-of-$q$, level-of-$r$) combination. Some pairs (e.g., Worker–Year) may be nearly dense when both factors interact freely, but in that case one dimension is typically much smaller than the other ($m_{\text{Year}} \ll m_{\text{Worker}}$), keeping the block cheap to work with.

---

## 4. Classical Solution: Alternating Projection

### 4.1 Three views of one algorithm

The classical method sweeps through factors one at a time, updating each factor's coefficients while holding the others fixed. This single operation has three equivalent descriptions that reveal different aspects of its structure.

---

#### View 1: Observation-space demeaning

Each factor $q$ has an $n \times m_q$ indicator matrix $D_q^{\text{obs}}$ (the $q$-th column block of $D$, with $D_q^{\text{obs}}[i,j] = 1$ iff $f_q(i) = j$). Given the current fitted values $\hat{y} = \sum_r D_r^{\text{obs}} \alpha_r$, the factor-$q$ update computes the partial residual $e = y - \sum_{r \neq q} D_r^{\text{obs}} \alpha_r$ and replaces $\alpha_q$ with the $W$-weighted mean of $e$ within each level:

$$
\alpha_{q,j} \leftarrow \frac{\sum_{i:\, f_q(i)=j} w_i\, e_i}{\sum_{i:\, f_q(i)=j} w_i}
$$

This is **demeaning by factor $q$**: each coefficient becomes the weighted average of the residual at its level.

---

#### View 2: Coefficient-space block Gauss-Seidel

The demeaning formula can be rewritten purely in terms of the Gramian blocks, eliminating the observation-level sums. Split the numerator by expanding $e_i = y_i - \sum_{r \neq q} \alpha_{r,f_r(i)}$:

$$
\sum_{i:\, f_q(i)=j} w_i\, e_i
= \underbrace{\sum_{i:\, f_q(i)=j} w_i\, y_i}_{= \; b_{q,j}}
\;-\; \sum_{r \neq q} \underbrace{\sum_{i:\, f_q(i)=j} w_i\, \alpha_{r,f_r(i)}}_{= \; \sum_\ell\, C_{qr}[j,\ell]\;\alpha_{r,\ell}}
$$

The first term is $b_{q,j} = (D_q^{\text{obs}\top} W y)_j$, the $q$-th block of the right-hand side. The second term groups observations by their level $\ell$ in factor $r$: each group contributes $C_{qr}[j,\ell] = \sum_{i:\, f_q(i)=j,\, f_r(i)=\ell} w_i$ times $\alpha_{r,\ell}$. The denominator is $D_q[j,j] = \sum_{i: f_q(i)=j} w_i$. So the weighted mean becomes:

$$
\alpha_{q,j} \leftarrow \frac{1}{D_q[j,j]} \left( b_{q,j} - \sum_{r \neq q} \sum_{\ell} C_{qr}[j,\ell]\;\alpha_{r,\ell}^{(\cdot)} \right)
$$

or in matrix form, for block $q$ processed in order $q = 1, 2, \ldots, Q$:

$$
D_q\,\alpha_q^{(k+1)} = b_q - \sum_{r < q} C_{qr}\,\alpha_r^{(k+1)} - \sum_{r > q} C_{qr}\,\alpha_r^{(k)}
$$

Factors $r < q$ have already been updated in this sweep (use $\alpha_r^{(k+1)}$); factors $r > q$ have not (use $\alpha_r^{(k)}$). This is exactly **block Gauss-Seidel**. Decomposing $G = \mathcal{D} + \mathcal{L} + \mathcal{U}$ into the block diagonal $\mathcal{D} = \text{blkdiag}(D_1, \ldots, D_Q)$, the strictly lower block-triangular $\mathcal{L}$ (blocks $C_{qr}$ with $r < q$), and the strictly upper block-triangular $\mathcal{U}$ (blocks $C_{qr}$ with $r > q$), the full sweep is:

$$
(\mathcal{D} + \mathcal{L})\,\alpha^{(k+1)} = b - \mathcal{U}\,\alpha^{(k)}
$$

The left-hand side couples to the already-updated factors; the right-hand side uses the stale values. The observation-level sums have been absorbed into the Gramian blocks $D_q$ and $C_{qr}$ — the algorithm now operates entirely in coefficient space $\mathbb{R}^m$.

---

#### View 3: Alternating projection

The column space $V_q = \text{col}(D_q^{\text{obs}}) \subset \mathbb{R}^n$ consists of all observation-space vectors that are constant within each level of factor $q$:

$$
V_q = \bigl\{\, v \in \mathbb{R}^n : f_q(i) = f_q(j) \;\Longrightarrow\; v_i = v_j \bigr\}
= \bigl\{\, D_q^{\text{obs}}\,\alpha_q : \alpha_q \in \mathbb{R}^{m_q} \bigr\}.
$$

Let $P_q$ denote the $W$-orthogonal projection onto $V_q$:

$$
P_q = D_q^{\text{obs}}\; D_q^{+}\; D_q^{\text{obs}\top} W
$$

The complementary projection $I - P_q$ maps onto the orthogonal complement $V_q^\perp$. This is precisely **demeaning**: subtracting weighted group means removes the component in $V_q$, leaving only what is $W$-orthogonal to factor $q$.

After updating factor $q$ in the sweep, the overall residual $r = y - \sum_s D_s^{\text{obs}}\,\alpha_s$ satisfies $P_q\, r = 0$, i.e., $r \in V_q^\perp$. At convergence the residual lies in the orthogonal complement of every factor simultaneously:

$$
r^* \;\in\; V_1^\perp \cap V_2^\perp \cap \cdots \cap V_Q^\perp
\;=\; \bigl(V_1 + V_2 + \cdots + V_Q\bigr)^\perp.
$$

Therefore the fitted value $\hat{y} = y - r^*$ equals $P_{V_1 + \cdots + V_Q}\, y$, the $W$-orthogonal projection of $y$ onto the sum of column spaces — exactly the least-squares solution of the fixed-effects model.

Each full sweep applies the product of projections $(I - P_Q) \cdots (I - P_2)(I - P_1)$ to the residual. By the Halperin extension of von Neumann's theorem, iterating this product converges to $P_{\bigcap_q V_q^\perp}$, confirming the residual contracts toward $(V_1 + \cdots + V_Q)^\perp$.

---

These are not three algorithms — they are three notations for the same computation:

| Viewpoint | Space | One factor-$q$ step | Key object |
|---|---|---|---|
| Demeaning | $\mathbb{R}^n$ (observations) | weighted mean of partial residual | raw data $(y, w)$ |
| Block GS | $\mathbb{R}^m$ (coefficients) | $D_q^{-1}(b_q - \sum C_{qr}\alpha_r)$ | Gramian blocks $(D_q, C_{qr})$ |
| Alternating proj. | $\mathbb{R}^n$ (observations) | project residual onto $V_q^\perp$ via $(I - P_q)$ | orthogonal complements $V_q^\perp$ |


### 4.2 The running example

Continuing the Worker/Firm/Year example, one full sweep processes $Q = 3$ factors in order:

**Step 1** — Update Worker ($q = 1$), holding Firm and Year fixed at $\alpha_F^{(k)}$, $\alpha_Y^{(k)}$:

$$
\alpha_{\text{W1}}^{(k+1)} = \frac{1}{2}\bigl(b_{\text{W1}} - 1 \cdot \alpha_{\text{F1}}^{(k)} - 1 \cdot \alpha_{\text{F2}}^{(k)} - 1 \cdot \alpha_{\text{Y1}}^{(k)} - 1 \cdot \alpha_{\text{Y2}}^{(k)}\bigr)
$$

Each worker's coefficient becomes the weighted average of $(y_i - \text{firm effect} - \text{year effect})$ for observations at that worker.

**Step 2** — Update Firm ($q = 2$), using **updated** Worker values $\alpha_W^{(k+1)}$ but **stale** Year values $\alpha_Y^{(k)}$:

$$
\alpha_{\text{F1}}^{(k+1)} = \frac{1}{3}\bigl(b_{\text{F1}} - 1 \cdot \alpha_{\text{W1}}^{(k+1)} - 2 \cdot \alpha_{\text{W2}}^{(k+1)} - 2 \cdot \alpha_{\text{Y1}}^{(k)} - 1 \cdot \alpha_{\text{Y2}}^{(k)}\bigr)
$$

Each firm's coefficient becomes the weighted average of $(y_i - \text{worker effect} - \text{year effect})$, but the year effects are still from the previous iteration.

**Step 3** — Update Year ($q = 3$), using updated Worker $\alpha_W^{(k+1)}$ and updated Firm $\alpha_F^{(k+1)}$:

$$
\alpha_{\text{Y1}}^{(k+1)} = \frac{1}{3}\bigl(b_{\text{Y1}} - 1 \cdot \alpha_{\text{W1}}^{(k+1)} - 1 \cdot \alpha_{\text{W2}}^{(k+1)} - 1 \cdot \alpha_{\text{W3}}^{(k+1)} - 2 \cdot \alpha_{\text{F1}}^{(k+1)} - 1 \cdot \alpha_{\text{F2}}^{(k+1)}\bigr)
$$

Only the last factor in the sweep sees fully updated values from all other factors. Workers were updated with stale firm and year effects; firms were updated with stale year effects. With $Q = 3$, two out of three updates use partially stale information — illustrating the degradation described in Section 4.4.

Repeat until convergence.

### 4.3 Convergence rate and the Friedrichs angle

Since each sweep is alternating projection between the orthogonal complements $V_1^\perp, \ldots, V_Q^\perp$ (Section 4.1), the convergence rate is governed by the **Friedrichs angle** $\theta_F$ between the column spaces. Informally, $\theta_F$ measures how close to collinear two factor subspaces are after removing their trivial intersection (constant shifts within connected components). For $Q = 2$, the error contracts by $\cos^2(\theta_F)$ per full sweep:

$$
\|r^{(k)}\|_W \leq \cos^{2k}(\theta_F)\;\|r^{(0)}\|_W
$$

- **High mobility** (workers move between many firms): $\cos(\theta_F)$ is small, convergence is fast.
- **Low mobility** (workers stuck at one firm): $\cos(\theta_F) \to 1$, convergence degrades sharply.

For $Q > 2$, any near-collinear pair bottlenecks the entire iteration.


### 4.4 Limitations

Three structural limitations of alternating projection follow from this analysis:

1. **No way to improve the local solve** — each block solve is already exact ($D_q$ is diagonal), so there is no knob to turn within a subdomain. The only option is to iterate more.

2. **Cross-factor structure is ignored** — the local solve for factor $q$ knows nothing about the coupling $C_{qr}$. When $\cos(\theta_F)$ is close to 1, the coupling is strong and the projections make little progress — yet the algorithm cannot exploit the cross-tabulation structure that causes the difficulty.

3. **Degradation with more factors** — for $Q > 2$, each factor's update uses stale values from factors not yet processed in the current sweep. The effective convergence rate worsens as the number of interacting pairs grows.

---

## 5. The Domain Decomposition Perspective

### 5.1 Block Gauss-Seidel is a domain decomposition method

Block Gauss-Seidel partitions the coefficient vector $\alpha \in \mathbb{R}^m$ into $Q$ blocks — one per factor — and sweeps through them sequentially. Each step solves a small system (the diagonal block $D_q$) using the current residual restricted to that block's DOFs.

This is exactly a **multiplicative Schwarz** method. The coefficient space $\mathbb{R}^m$ is decomposed into $Q$ non-overlapping subspaces $S_q \subset \mathbb{R}^m$, where $S_q$ contains the DOFs of factor $q$. The restriction operator $R_q: \mathbb{R}^m \to S_q$ extracts the factor-$q$ coefficients, and the local operator $A_q = R_q\, G\, R_q^\top = D_q$ is the corresponding diagonal block.

Each multiplicative Schwarz correction computes a local solve and injects the result back:

$$
\alpha \;\leftarrow\; \alpha + R_q^\top\, A_q^+\, R_q\, r_{\text{current}}
$$

where $r_{\text{current}} = b - G\,\alpha$ is the residual at the time factor $q$ is processed. This is precisely the block GS update from Section 4.1 (View 2).

**Notation.** The Schwarz subspaces $S_q \subset \mathbb{R}^m$ live in coefficient space. They are related to — but distinct from — the column spaces $V_q = \text{col}(D_q^{\text{obs}}) \subset \mathbb{R}^n$ from Section 4.1 (View 3), which live in observation space. The design matrix $D_q^{\text{obs}}$ maps $S_q \to V_q$ (coefficients to observations); its adjoint $D_q^{\text{obs}\top} W$ maps back $\mathbb{R}^n \to S_q$ (aggregating observations into levels). The alternating-projection view (cycling through $V_q^\perp$ in $\mathbb{R}^n$) and the Schwarz view (cycling through $S_q$ in $\mathbb{R}^m$) describe the same algorithm from different spaces.

| Schwarz concept | Block GS instantiation |
|---|---|
| Number of subdomains | $Q$ (one per factor) |
| Subdomain $S_q$ | DOFs of factor $q$: indices $\{m_1 + \cdots + m_{q-1} + 1, \;\dots,\; m_1 + \cdots + m_q\}$ |
| Overlap | **None** — subdomains partition $\mathbb{R}^m$ |
| Local operator $A_q$ | $D_q$ (diagonal) |
| Local solve cost | $O(m_q)$ — just division |
| Partition-of-unity weights | All 1 (no overlap) |

The decomposition has $Q$ small, cheap subdomains that cover all DOFs exactly once:

![Factor-level decomposition](images/graph_factor_level.svg)

The same interaction graph from Section 3.2, now grouped into factor-level subdomains. Every edge crosses a subdomain boundary — the local operators $D_q$ are trivially diagonal, with no internal edges. The local solves are trivial but they capture **none** of the cross-factor coupling.

### 5.2 The general Schwarz framework

The block GS / factor-level decomposition is the simplest instance of a broader family. A **Schwarz method** decomposes $\mathbb{R}^m$ into subspaces $S_1, \ldots, S_N$ (possibly overlapping), with restriction operators $R_i: \mathbb{R}^m \to S_i$ and local operators $A_i = R_i\, G\, R_i^\top$:

- **Multiplicative Schwarz** (sequential): process subdomains one by one, updating the global residual after each correction $\alpha \leftarrow \alpha + R_i^\top A_i^+ R_i\, r_{\text{current}}$.

- **Additive Schwarz** (parallel): apply all local solves to the same residual and sum the weighted corrections:
$$
M^{-1} r = \sum_i R_i^\top \tilde{D}_i\, A_i^+\, \tilde{D}_i\, R_i\, r
$$
where $\tilde{D}_i$ are partition-of-unity weights that prevent double-counting when subdomains overlap.

The choice of subdomains determines both the cost per iteration and the quality of the preconditioner. Factor-level subdomains (§5.1) are cheap but weak. Richer subdomains can capture more of the Gramian's structure.

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
| Local operator | $D_q$ (diagonal, trivial) | $G_{qr}$ (bipartite, Laplacian after sign-flip) |
| Cross-factor coupling | Ignored | Fully captured per pair |
| Local solve cost | $O(m_q)$ | $O(m_q + m_r)$ to $O((m_q + m_r) \log(m_q + m_r))$ |

The local systems are now proper sparse linear systems — no longer trivially diagonal — but after a sign-flip transformation they become graph Laplacians, for which nearly-linear-time solvers exist. The trade-off: harder local solves in exchange for dramatically fewer outer iterations, because the preconditioner captures far more of the global Gramian's structure.

### 5.4 Continuing the example

The Worker/Firm/Year example ($Q = 3$) produces $\binom{3}{2} = 3$ factor-pair subdomains:

| Subdomain | Factor pair | DOFs | Size |
|-----------|-------------|------|------|
| 1 | (Worker, Firm) | W1, W2, W3, F1, F2 | 5 |
| 2 | (Worker, Year) | W1, W2, W3, Y1, Y2 | 5 |
| 3 | (Firm, Year) | F1, F2, Y1, Y2 | 4 |

Each DOF appears in $Q - 1 = 2$ subdomains, requiring partition-of-unity weights of $1/\sqrt{2}$ per side (see [Part 2, Section 4.2](2_solver_architecture.md#42-partition-of-unity)).

The same interaction graph, now with three overlapping factor-pair subdomains drawn around it:

![Factor-pair decomposition](images/graph_factor_pair.svg)

The Worker–Firm subdomain (red) covers the top two rows. The Firm–Year subdomain (blue) covers the bottom two. The Worker–Year subdomain (green) wraps around the Firms in a C-shape — it contains Workers and Years but not Firms. Every edge is now *inside* a subdomain. Each node sits in exactly 2 of the 3 boxes (e.g., Workers are in red and green), which is why partition-of-unity weights are needed.

For instance, subdomain 1 (Worker–Firm) solves:

$$
A_{\text{WF}} = \begin{pmatrix} D_W & C_{WF} \\ C_{WF}^\top & D_F \end{pmatrix}
= \left(\begin{array}{ccc|cc}
2 & 0 & 0 & 1 & 1 \\
0 & 2 & 0 & 2 & 0 \\
0 & 0 & 2 & 0 & 2 \\
\hline
1 & 2 & 0 & 3 & 0 \\
1 & 0 & 2 & 0 & 3
\end{array}\right)
$$

This is a proper linear system (not just a diagonal), capturing how workers and firms interact. With a good approximate factorization, three such local solves cover all pairwise interactions — far more information per iteration than the factor-level decomposition's three diagonal solves.

### 5.5 Where this leads

The factor-pair decomposition raises three algorithmic questions:

1. **How to solve the local Laplacian systems efficiently?** → Approximate Cholesky factorization with Schur complement reduction ([Part 3](3_local_solvers.md))
2. **How to combine the local corrections?** → Additive or multiplicative Schwarz with partition-of-unity weights ([Part 2](2_solver_architecture.md))
3. **How to drive the global iteration?** → Preconditioned CG or GMRES ([Part 2](2_solver_architecture.md))

---

## References

**Correia, S.** (2016). *A Feasible Estimator for Linear Models with Multi-Way Fixed Effects*. Working paper. Describes the fixed-effects normal equations, their block structure, and iterative solution via alternating projections.

**Xu, J.** (1992). *Iterative Methods by Space Decomposition and Subspace Correction*. SIAM Review, 34(4), 581–613. Provides the abstract space decomposition framework for additive and multiplicative Schwarz methods.

**Toselli, A. & Widlund, O. B.** (2005). *Domain Decomposition Methods — Algorithms and Theory*. Springer. Comprehensive reference for the theory and convergence analysis of Schwarz methods.
