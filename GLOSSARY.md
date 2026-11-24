# Glossary

Terms the code and docs use. Where the literature differs, the entry says which meaning we
take. "(not: …)" lists synonyms we don't use.

## Symbols

| Symbol | Meaning |
|---|---|
| $n$, $i$ | number of observations, an observation |
| $t$, $T$ | a term, the number of terms |
| $\ell_t(i)$, $L_t$ | level of observation $i$ in term $t$, number of levels of term $t$ |
| $z_{t,j}$ | slope covariate $j$ of term $t$ |
| $w_i$, $W = \operatorname{diag}(w)$, $w_\ell$ | weight, weight matrix, level weight |
| $D$, $D_t$, $E_t$ | design, term block, incidence |
| $m$ | number of DOFs (columns of $D$) |
| $y$, $\beta$, $\tilde y$ | response, coefficients, demeaned response |
| $R_t$, $\tilde D = DR$, $\hat x$ | orthogonalizing change of basis, orthogonal design, solve coordinates |
| $u_q(i)$ | orthogonalized loading of channel $q$ at observation $i$ |
| $A = W^{1/2} \tilde D$, $G = A^\top A$ | weighted design, full Gramian |
| $e$, $A_e$, $A_k$, $P_e$ | eliminated term, its block of $A$, the kept terms' block, projection onto $\operatorname{range} A_e$ |
| $A_r = (I - P_e) A_k$, $b = (I - P_e) W^{1/2} y$ | system matrix, right-hand side |
| $S = A_r^\top A_r$ | Gramian |
| $\Delta_q$, $C_{qr}$, $G_{qr}$ | diagonal block of channel $q$, cross block and pair block of channels $q, r$ |
| $V$ | Gram factor of a pair block, $G_{qr} = VV^\top$ |
| $M$ | preconditioner, $M \approx S$ |
| $R_{qr}$, $\Omega_{qr}$ | restriction and partition-of-unity weights of the subdomain of channels $q, r$ |
| $c_j$, $\omega_j$ | overlap count and partition-of-unity weight of DOF $j$ |
| $B_{qr}$ | local solve, the approximate inverse of $G_{qr}$ |
| $\Lambda$ | dominance scaling |
| $\sigma$ | signature |
| $x_k$, $r_k = b - A_r x_k$ | iterate, residual |
| $\rho_k$ | contraction ratio |

## Panel and design

- **Observation** $i$: one record of the panel; it gives one row of $D$.
- **Factor:** a categorical variable that classifies observations. Its values are **levels**.
- **Label:** the caller's `u32` code for a level. Labels need not be contiguous; results are
  addressed by label.
- **Position:** a level's compact index $0, \dots, L_t - 1$, over the levels that occur.
- **Term** (API: `Effect`): a factor together with its coefficient columns, an optional
  intercept and zero or more slopes. Each term has its own factor; "cross-factor" means
  "between two different terms". A design built from a categories array has one intercept-only
  term per factor.
- **Intercept:** the column whose loading is $1$: the fixed effect of the level.
- **Slope** (not: interaction): the column whose loading is a covariate $z_{t,j}$, giving each
  level its own coefficient on $z_{t,j}$ (fixest's `f[z]`).
- **Loading:** the value a column multiplies at observation $i$: $1$ for the intercept,
  $z_{t,j}(i)$ for slope $j$.
- **Design** $D$: the $n \times m$ matrix $[D_1 \cdots D_T]$. Row $i$ of the **term block**
  $D_t$ holds the loadings $[1?, z_{t,1}(i), \dots]$ in the columns of level $\ell_t(i)$, and
  zeros elsewhere. Written $D = E \bullet X$: the incidence picks the level, the loadings $X$
  fill it. The design carries no weights.
- **Incidence** $E_t$: the $n \times L_t$ 0/1 matrix with $E_t[i, \ell_t(i)] = 1$.
- **Segment** $R_\ell$: the rows of level $\ell$ of the term the rows are sorted by, which are
  contiguous, so its incidence is $\operatorname{blockdiag}(\mathbb{1}_{R_0}, \dots)$. A
  **segmented** pass works on each segment alone, or sums over them.
- **Channel:** one coefficient column of a term, the intercept or one slope, addressed
  (term, column) with the intercept first. It spans $L_t$ DOFs, one per level.
- **DOF** (degree of freedom; not: factor level): one coefficient, a channel at a level; one
  column of $D$. Only in an intercept-only design is a DOF a level.
- **Coefficient layout:** the terms' blocks back to back in term order; within term $t$'s block,
  column $c$ of the level at position $p$ sits at $c L_t + p$. A **coefficient address** is
  (channel, label).
- **Weight** $w_i$: finite and $\ge 0$; unweighted means $W = I$. A zero weight is the same as
  removing the observation from the fit.
- **Level weight** $w_\ell$: the sum of the weights of the level's observations. A
  **weightless** level has $w_\ell = 0$.
- **Caller order / internal order:** the observation order the caller passed, and the order the
  solver stores rows in. Every per-observation result is in caller order.

## The problem

- **Response** $y$ (not: target): one column to project, $y$ or a column of $X$ in the model
  $y = X\beta_X + D\beta + \varepsilon$. A **batch** is several responses on one design and
  weighting.
- **FWL** (Frisch–Waugh–Lovell): $\beta_X$ is the regression of the demeaned $y$ on the
  demeaned columns of $X$. `within` computes the demeaning.
- **Normal equations:** $D^\top W D \beta = D^\top W y$, the optimality condition of
  $\min_\beta \lVert W^{1/2}(y - D\beta) \rVert_2$. They are always consistent.
- **Fit** $D\hat\beta$: the $W$-orthogonal projection of $y$ onto the range of $D$; unique on
  observations with $w_i > 0$, whichever solution $\hat\beta$ is taken.
- **Demeaned response** $\tilde y = y - D\hat\beta$ (not: residual, which names the solver's
  quantity below). With a single intercept-only term it is $y$ minus its level means.
- **Interaction graph:** vertices are the levels of all terms; levels of two different terms
  share an edge when some observation with $w_i > 0$ has both.
- **Connected component:** a maximal set of levels joined by paths in the interaction graph.
- **Mobility:** how densely observations link the levels of two factors. In a worker–firm
  panel a **mover** is a worker observed at two or more firms, a **stayer** at one; few movers
  is low mobility, the hard case.
- **Identified:** a direction of $\beta$ that changes the fit. In an intercept-only design each
  connected component leaves (participating terms $-1$) unidentified shifts of a constant
  between terms; slopes add more only through collinear loadings within a level.
- **Unidentified direction:** a per-level DOF the data cannot identify, reported to the caller:
  the intercept of a weightless level, or a **dropped slope**, one whose loadings within its
  level are collinear with the intercept and the earlier slopes. Its coefficient is $0$, never
  NaN.
- **Correct mod identification:** $\hat\beta$ solves the normal equations. Solutions differ only
  by unidentified directions, so this is the correctness criterion; $\hat\beta$ itself can
  differ from tools that drop a reference level.

## Solve coordinates

- **Orthogonalization:** per level of each slope-bearing term, the slopes are centered on their
  weighted level mean (when the term has an intercept) and made $W$-orthonormal by pivoted
  Gram–Schmidt. $R_t$ is the change of basis, the **orthogonal design** $\tilde D = DR$, and
  $\tilde D_t^\top W \tilde D_t$ is diagonal: $w_\ell$ on the intercept, $1$ on kept slopes,
  $0$ on dropped ones.
- **Solve coordinates** $\hat x$: coefficients of $\tilde D$. The **back-transform**
  $\beta = R\hat x$ returns to the caller's parametrization; $\tilde D\hat x = D\beta$.
- **Weighted design** $A = W^{1/2}\tilde D$, $n \times m$, and the **full Gramian**
  $G = A^\top A = \tilde D^\top W \tilde D$ (not: normal matrix, Hessian), symmetric PSD. Neither
  is assembled; the solve works on their reduced forms below.
- **Eliminated term** $e$: the design's **sort term**, the one the rows are sorted by and with the
  most DOFs. Its block $G_{ee}$ is diagonal, so the projection $P_e = A_e G_{ee}^{+} A_e^\top$
  onto its span acts on each segment alone. The other terms are **kept**; their coefficients, the
  **unknowns** $x$, are what the iteration solves for, $m_k$ of them.
- **System matrix** $A_r = (I - P_e) A_k$ (not: operator), $n \times m_k$, applied, never
  assembled. LSMR iterates on $A_r$ with the **right-hand side** $b = (I - P_e) W^{1/2} y$.
  **Recovery** then gives $x_e = G_{ee}^{+} A_e^\top (W^{1/2} y - A_k x)$.
- **Gramian** $S = A_r^\top A_r = G_{kk} - G_{ke} G_{ee}^{+} G_{ek}$, the Schur complement of
  $G_{ee}$ in $G$; $m_k \times m_k$, symmetric PSD. Preconditioners are built from it, read off
  $G$'s pair factors; the iteration never sees it.
- **Diagonal block** $\Delta_q$ of channel $q$: diagonal, $w_\ell$ or $1$ or $0$ per level as
  above. Two channels of the same term are orthogonal, so their block of $G$ is $0$.
- **Cross block** $C_{qr}$ of channels $q, r$ of different terms:
  $C_{qr}[j, k] = \sum_{i:\, \ell_q(i) = j,\, \ell_r(i) = k} w_i\, u_q(i)\, u_r(i)$. For two
  intercepts it is the weighted **cross-tabulation**, $\ge 0$; with a slope it is signed.

## Preconditioning

- **Preconditioner** $M \approx S$: a symmetric PSD $m_k \times m_k$ operator; LSMR applies
  $M^{-1}$ once per iteration.
- **Identity** (API: `Off`): $M = I$, unpreconditioned.
- **Diagonal** (not: Jacobi in prose): $M = \operatorname{diag}(S)$; a zero diagonal entry is
  inverted to $0$.
- **Channel pair** (not: factor pair): two channels of different terms; $T$ terms with $c_t$
  channels give $\sum_{t < u} c_t c_u$ pairs. A **factor pair** is the channel pair of two
  intercepts, the only kind in an intercept-only design.
- **Plain pair / slope pair:** a pair of two intercepts / a pair with at least one slope.
- **Pair block** $G_{qr} = \begin{pmatrix} \Delta_q & C_{qr} \\ C_{qr}^\top & \Delta_r \end{pmatrix}$:
  the principal submatrix of $G$ on the pair's DOFs.
- **Gram factor** $V$ of a pair block: $G_{qr} = VV^\top$ with one column per observation,
  $\sqrt{w_i}\,(u_q(i)\, e_{\ell_q(i)} + u_r(i)\, e_{L_q + \ell_r(i)})$: two nonzeros per
  column (**bipartite factor-width-2 form**).
- **Subdomain:** the unknowns of one channel pair; its **local operator** approximates $S$ there.
  Two kept channels get $G_{qr}$, which is $S$'s block without $e$'s Schur term
  $G_{qe} G_{ee}^{+} G_{er}$. A kept channel $q$ with one of $e$'s, $c$, gets $G_{qc}$'s Schur
  complement onto $q$: $\Delta_q$ less $c$'s share of that term.
- **Restriction** $R_{qr}$: picks the subdomain's DOFs from a global vector; its transpose, the
  **prolongation**, adds them back.
- **Overlap count** $c_j$: the number of subdomains containing DOF $j$.
- **Partition-of-unity weights** $\omega_j = 1/\sqrt{c_j}$, $\Omega_{qr}$ their diagonal on the
  subdomain, so that $\sum R_{qr}^\top \Omega_{qr}^2 R_{qr} = I$ on covered DOFs. Used in intercept-only
  designs; with any slope channel every weight is $1$.
- **Local solve** $B_{qr}$: the approximate inverse of the pair block, block diagonal over the
  pair's components.
- **Additive Schwarz:** $M^{-1} = \sum_{(q,r)} R_{qr}^\top \Omega_{qr} B_{qr} \Omega_{qr} R_{qr}$; one-level (no
  coarse space), local solves independent and in parallel.
- **MAP** (method of alternating projections; not: zig-zag, demeaning algorithm): updates each
  factor's coefficients to the weighted level means of the partial residual, one factor at a
  time. It is block Gauss–Seidel on the normal equations, i.e. multiplicative Schwarz with one
  subdomain per factor, which `within` replaces.

## Local solver

approx-chol computes each local solve; these are the terms its configuration names.

- **Component:** a connected component of the pair block's graph. A level with no cross entry is
  a one-level component; with a zero diagonal it solves to $0$.
- **Balanced / frustrated:** whether some signature $\sigma \in \{\pm 1\}^n$ makes every
  off-diagonal entry of $\sigma G_{qr} \sigma$ nonpositive. Plain pairs are always balanced; slope
  pairs generically are not, and a frustrated component is factored through its Gremban cover,
  a system of twice the size.
- **Dominance scaling:** a positive diagonal $\Lambda$ with $\Lambda G_{qr} \Lambda$ weakly diagonally
  dominant (each diagonal entry at least the sum of its row's off-diagonal magnitudes). It is
  computed per component within `max_iterations` and **certified** if it meets dominance within
  `tolerance`.
- **Unscalable component:** one whose scaling is not certified. Its diagonal is clamped up to
  dominance, which costs iterations, never correctness; `on_failure` makes it a warning or an
  error.
- **Floating:** a balanced component whose scaled block has every row sum $0$. It is singular,
  and its local solve has the same kernel.
- **Schur reduction:** eliminates the pair's larger channel, whose block is diagonal, and solves
  on the other. **Exact** keeps each eliminated level's clique of fill edges; **approximate**
  replaces it by a random spanning tree of the clique (a **clique tree**), $d - 1$ edges instead
  of $d(d-1)/2$, with the clique as expectation. The **split** $k$ divides each edge into $k$
  copies before sampling: denser, more accurate.
- **Approximate Cholesky** (AC($k$)): factors the reduced system by elimination in random order,
  each clique replaced by a clique tree; `split_merge` is its $k$.
- **Dense threshold:** the largest block factored exactly by dense Cholesky instead, a Gremban
  cover counting both halves; $0$ disables it.
- **Seed:** initializes the random choices of both samplers.

## Iteration

- **LSMR** (Fong, Saunders): a Krylov method for $\min \lVert b - A_r x \rVert_2$ built on
  Golub–Kahan bidiagonalization of $A_r$; $\lVert A_r^\top r_k \rVert$ decreases monotonically.
- **Modified LSMR** (not: MLSMR in prose): preconditioned LSMR, bidiagonalizing in the
  $M^{-1}$ inner product (modified Golub–Kahan) with one $M^{-1}$ application per iteration.
- **Iteration** (not: sweep, which is MAP's): one step of the bidiagonalization and its update
  of $x_k$.
- **Residual** $r_k = b - A_r x_k$; the **normal-equation residual** is $A_r^\top r_k$.
- **Relative normal-equation residual** (API: `residual`): $\lVert A_r^\top r_k \rVert /
  \lVert A_r^\top b \rVert$, in $M$'s metric when preconditioned.
- **Tolerance stop:** the recurrence's estimate meets `tol`, either for the least-squares
  residual or for the backward error $\lVert A_r^\top r_k \rVert / (\lVert A_r \rVert \lVert r_k \rVert)$.
- **Exit audit:** at a tolerance stop, the residual and the unpreconditioned gradient
  $A_r^\top r_k$ are recomputed. A **refuted** stop restarts from $x_k$; a stop that survives is
  **converged**.
- **False convergence:** a small stopping quantity with a large true one, e.g. when $\ker M$
  meets $\operatorname{range} A_r^\top$, where the $M$-metric gradient vanishes but the true one
  does not.
- **Reorthogonalization window** (API: `local_size`): the number of recent basis vectors each
  new one is reorthogonalized against.
- **Warm start:** an iteration started from a given $x_0$ instead of $0$.

## Adaptive ladder

- **Ladder** (API: `Adaptive`, the default): preconditioners tried in order, each a **rung**:
  diagonal, then additive Schwarz.
- **Contraction ratio** $\rho_k$: the relative normal-equation residual at iteration $k$ over the
  one at $k - 1$.
- **Stall:** `window` consecutive contraction ratios above `threshold` (defaults 4 and 0.7).
- **Escalation:** on the first stall, the Schwarz rung is built once for the solver and every
  stalled response resumes from its iterate on it. The ladder is then **settled**: later solves
  start on Schwarz. Reported iterations sum over both rungs.
