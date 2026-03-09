# Part 4: Implementation Notes

This is Part 4 of the algorithm documentation for the `within` solver. It provides Rust-level pseudocode and references specific source files for each major algorithmic component.

**Series overview**:
- [Part 1: Fixed Effects and Block Iterative Methods](1_fixed_effects_and_block_methods.md)
- [Part 2: Preconditioned Krylov Solvers and Schwarz Decomposition](2_solver_architecture.md)
- [Part 3: Local Solvers and Approximate Cholesky](3_local_solvers.md)
- **Part 4: Implementation Notes** (this document)

**Prerequisites**: Parts 1–3 (algorithmic understanding). This section is intended for contributors working with the Rust codebase.

---

## 1. Fused Observation Scan

The setup phase can build both the subdomain decomposition and the explicit Gramian CSR from a single observation scan, avoiding redundant passes over the data.

**Source**: `crates/within/src/operator/preconditioner.rs`

```
function build_preconditioner_fused(design, config):
    // Single call builds both subdomain data and Gramian block data
    (domains, pair_blocks) = build_domains_and_gramian_blocks(design)

    // Assemble explicit Gramian from blocks (no observation re-scan)
    gramian = Gramian::from_pair_blocks(pair_blocks, factors, n_dofs)

    // Build preconditioner from pre-built domains
    preconditioner = build_schwarz(FromParts(domains), config)

    return (gramian, preconditioner)
```

The key function `build_domains_and_gramian_blocks` processes each factor pair in parallel:

**Source**: `crates/within/src/domain/factor_pairs.rs`

```
function build_domains_and_gramian_blocks(design):
    all_active = find_all_active_levels(design)  // one O(nQ) scan
    pairs = [(q, r) for q < r]

    results = pairs.par_iter().flat_map(|(q, r)|:
        mapping = build_compact_mapping(all_active[q], all_active[r])
        cross_tab = Arc::new(accumulate_cross_block(design, q, r, mapping))

        // Split into components → subdomains
        components = cross_tab.bipartite_connected_components()
        subdomains = split_into_subdomains_arc(components, cross_tab.clone())

        // Also produce Gramian block data (shares the Arc<CrossTab>)
        block_data = PairBlockData { q, r, cross_tab, local_to_global }

        return (subdomains, block_data)
    )

    compute_partition_weights(subdomains, n_dofs)
    return (subdomains, pair_blocks)
```

---

## 2. Implicit Gramian Operator

The implicit operator computes $y = D^\top W (D x)$ without forming $G$.

**Source**: `crates/within/src/domain.rs`

**Gather** ($t = Dx$): for each observation $i$, accumulate the factor-level coefficients.

```
function gather_add(design, x, y):
    // y[i] += sum over factors q of x[offset[q] + level(i, q)]

    if n_rows <= 10_000:    // sequential
        for q in 0..Q:
            col = factor_column(q)  // contiguous u32 slice if available
            for i in 0..n_rows:
                y[i] += x[offset[q] + col[i]]
    else:                   // parallel, chunked
        y.par_chunks_mut(4096).for_each(|chunk|:
            for q in 0..Q:          // inner loop: factors (cache-hot)
                for i in chunk_range:
                    chunk[i] += x[offset[q] + col[i]]
        )
```

**Scatter** ($x = D^\top W r$): for each factor, accumulate weighted contributions.

```
function scatter_add(design, value_fn, output):
    for q in 0..Q:
        n_levels = factors[q].n_levels

        strategy = select_strategy(n_rows, n_levels):
            n_rows <= 10_000            → Sequential
            n_levels < 100_000          → Fold
            n_levels >= 100_000         → Atomic

        match strategy:
            Sequential:
                for i in 0..n_rows:
                    output[offset[q] + level(i,q)] += value_fn(i)

            Fold:
                // Per-thread accumulators, then reduce
                result = par_fold(
                    init: vec![0.0; n_levels],
                    fold: |acc, i| acc[level(i,q)] += value_fn(i),
                    reduce: |a, b| a[j] += b[j] for all j
                )
                output[offset[q]..] += result

            Atomic:
                // Shared AtomicF64 array
                atomic_buf = AtomicF64::from(output[offset[q]..])
                par_for_each(0..n_rows, |i|:
                    atomic_buf[level(i,q)].fetch_add(value_fn(i))
                )
                copy atomic_buf back to output
```

The `Fold` strategy is preferred for moderate-sized factors (< 100K levels) because each thread's accumulator fits in cache. The `Atomic` strategy avoids allocating per-thread buffers when the number of levels is large, relying on low CAS contention when many bins spread the writes.

---

## 3. Bipartite-to-Laplacian Sign-Flip

The local solve wrapper handles the sign convention and optional augmentation.

**Source**: `crates/within/src/operator/local_solver.rs`

```
function ApproxCholSolver::solve_local(rhs, sol):
    match strategy:
        Laplacian:
            // Pure Laplacian, no transform needed
            sol = factor.solve_in_place(rhs)
            return

        LaplacianGramian { first_block_size: fbs }:
            solve_n = n_local

        GramianAugmented { first_block_size: fbs }:
            solve_n = n_local + 1

        Sddm:
            solve_n = n_local + 1

    // Sign-flip: negate second block of rhs
    if fbs is set:
        rhs[fbs..n_local] *= -1

    // Augmentation: zero-pad for ground node
    if solve_n > n_local:
        rhs[n_local] = 0

    // Center: subtract mean for Laplacian compatibility
    subtract_mean(rhs[..solve_n])

    // Solve the SDDM system
    sol[..solve_n] = factor.solve_in_place(rhs[..solve_n])

    // Undo sign-flip on solution (negate second block)
    if fbs is set:
        sol[fbs..n_local] *= -1
```

For `BlockElimSolver` (Schur complement path), the flow is similar but includes the forward elimination and back-substitution steps:

```
function BlockElimSolver::solve_local(rhs, sol):
    n = n_q + n_r

    // 1. Sign-flip and center
    negate_block(rhs[..n], n_q)    // negate r-block
    subtract_mean(rhs, n)

    // 2. Form reduced RHS (eliminate larger block)
    if eliminate_q:
        scratch[..n_r] = rhs[n_q..n] + C^T · diag(1/D_q) · rhs[..n_q]
    else:
        scratch[..n_q] = rhs[..n_q] + C · diag(1/D_r) · rhs[n_q..n]

    // 3. Solve reduced system
    reduced_factor.solve_in_place(scratch[..n_keep])

    // 4. Back-substitute for eliminated block
    // sol_elim[i] = (1/D_elim[i]) · (rhs_elim[i] + C_e2k · sol_keep)
    backsub_block(sol_elim, rhs_elim, cross_matrix, inv_diag_elim, sol_keep)

    // 5. Re-center and undo sign-flip
    subtract_mean(sol, n)
    negate_block(sol[..n], n_q)    // negate r-block
```

---

## 4. CrossTab Representation

The bipartite block is stored without materializing the full symmetric matrix.

**Source**: `crates/within/src/operator/gramian/cross_tab.rs`

```
struct CrossTab {
    c:      CsrBlock,   // n_q × n_r cross-tabulation C
    ct:     CsrBlock,   // n_r × n_q precomputed transpose C^T
    diag_q: Vec<f64>,   // length n_q: D_q diagonal
    diag_r: Vec<f64>,   // length n_r: D_r diagonal
}
```

Key operations:

- **`to_sddm()`**: assembles the full SDDM matrix $L = [D_q, -C; -C^\top, D_r]$ as a CSR `SparseMatrix` of dimension $(n_q + n_r)$. Column order is guaranteed sorted: for q-rows, diagonal column $i < $ all C columns $n_q + k$; for r-rows, all $C^\top$ columns $< n_q$ come before diagonal column $n_q + i$.

- **`bipartite_connected_components()`**: iterative DFS using an explicit stack. Nodes $0 \ldots n_q - 1$ are q-nodes; nodes $n_q \ldots n_q + n_r - 1$ are r-nodes. Follows C edges from q-nodes and $C^\top$ edges from r-nodes. Component indices are sorted for deterministic ordering. Complexity: $O(n_q + n_r + \text{nnz}(C))$.

- **`extract_component(comp)`**: extracts a sub-CrossTab for one connected component by building reverse maps and filtering CSR rows.

Active-level handling uses compact mapping: a `u32` map from original level indices to compact 0-based indices, with `u32::MAX` as sentinel for inactive levels.

---

## 5. Explicit Gramian CSR Construction

**Source**: `crates/within/src/operator/gramian/explicit.rs`

Two construction paths:

**Full build** (`build_full_matrix`): scans all observations to build diagonal counts and per-pair cross-tabulation accumulators.

```
function build_full_matrix(design):
    if n_obs > 100_000:    // parallel path
        diag = Vec<AtomicF64>[n_dofs]      // shared, atomic accumulation
        pair_tables = per_thread Vec<PairAccumulator>

        par_chunks(observations, chunk_size).for_each(|chunk|:
            for obs in chunk:
                for q in 0..Q:
                    diag[offset[q] + level(obs,q)].fetch_add(weight(obs))
                for (q,r) in pairs:
                    pair_tables[pair_idx].add(level_q, level_r, weight)
        )

        // Merge per-thread pair tables sequentially
        for thread_tables in all_tables:
            for (idx, table) in thread_tables:
                global_tables[idx].merge_from(table)
    else:                  // sequential path
        // Single-pass accumulation into plain Vec<f64> + PairAccumulator

    // Two-pass CSR assembly
    build_symmetric_csr(emit_entries_fn)
```

**From blocks** (`compose_gramian_from_blocks`): assembles CSR from pre-built `PairBlockData` objects. Exploits the monotonic factor-offset invariant: for any row $g$ in factor $f$'s range, blocks where $f$ is the r-factor contribute columns below $f$'s offset, and blocks where $f$ is the q-factor contribute columns above. This guarantees column-sorted CSR without per-row sorting.

```
function compose_gramian_from_blocks(blocks, factors, n_dofs):
    // Pass 1: count NNZ per row
    for each active DOF g:
        nnz[g] = 1                          // diagonal
        for each block touching this factor:
            nnz[g] += CSR row length in C or C^T

    // Prefix sum → indptr

    // Pass 2: fill values
    for each factor f, for each active level:
        insert C^T entries (columns < offset[f])   // lower triangle
        insert diagonal entry                       // on diagonal
        insert C entries (columns > offset[f])      // upper triangle
```

**CSR row sorting** (in `build_symmetric_csr`): after filling, each row's entries are sorted by column index. Two strategies based on row length:
- $\leq 64$ entries: insertion sort (cache-friendly for small rows)
- $> 64$ entries: permutation sort via cycle-following with $O(\text{len})$ auxiliary memory

---

## 6. Additive Schwarz Parallelism

**Source**: `crates/schwarz-precond/src/schwarz.rs`, `crates/schwarz-precond/src/domain.rs`

The additive Schwarz accumulates subdomain contributions into a shared buffer using atomic f64 operations.

```
function SchwarzPreconditioner::apply(r, z):
    // 1. Acquire buffer from pool (or allocate new)
    accum = pool.pop() or new Vec<AtomicU64>[n_dofs]  // all zeros

    // 2. Parallel subdomain application
    subdomains.par_iter().for_each_init(
        || (vec![0.0; max_scratch], vec![0.0; max_scratch]),  // per-thread
        |(r_scratch, z_scratch), entry|:
            entry.restrict_weighted(r, r_scratch)      // gather + PoU weight
            entry.solve_local(r_scratch, z_scratch)    // local A_i^{-1}
            entry.prolongate_weighted_add_atomic(z_scratch, accum)  // scatter
    )

    // 3. Readout and reset (swap-to-zero)
    if n_dofs > 100_000:
        accum.par_chunks(4096).map(|chunk|:
            z[i] = f64::from_bits(chunk[i].swap(0, Relaxed))
        )
    else:
        for i in 0..n_dofs:
            z[i] = f64::from_bits(accum[i].swap(0, Relaxed))

    // 4. Return buffer to pool (max 4 sets)
    pool.push(accum) if pool.len() < 4
```

The **atomic f64 addition** uses a CAS retry loop on `AtomicU64`:

```
function atomic_f64_add(target: &AtomicU64, val: f64):
    old_bits = target.load(Relaxed)
    loop:
        new_val = f64::from_bits(old_bits) + val
        match target.compare_exchange_weak(old_bits, new_val.to_bits(), Relaxed, Relaxed):
            Ok(_)       → break
            Err(actual) → old_bits = actual  // retry
```

`compare_exchange_weak` (allowing spurious failure) is used for better performance on ARM. `Relaxed` ordering suffices because the buffer is zeroed before use and read after all threads complete.

The buffer pool (`Arc<Mutex<Vec<SchwarzBuffers>>>`) holds up to 4 buffer sets. Clones of the preconditioner share the pool, enabling concurrent Krylov solvers (e.g., batch mode) to reuse buffers without excessive allocation.

---

## 7. Observation-Space Residual Updater

For multiplicative Schwarz, the residual must be updated after each subdomain correction. The observation-space updater avoids a full operator application by exploiting the sparse structure.

**Source**: `crates/within/src/operator/schwarz.rs` (the `ObservationSpaceUpdater` in `crates/within/src/operator/residual_update.rs`)

**Inverted index** (`DofObservationIndex`): precomputed in two passes. For any DOF $d$, `obs_for_dof(d)` returns the list of observations that touch $d$.

```
function DofObservationIndex::build(design):
    // Pass 1: count observations per DOF
    counts = [0u32; n_dofs]
    for obs in 0..n_obs:
        for q in 0..Q:
            counts[offset[q] + level(obs, q)] += 1

    // Prefix sum → offsets
    // Pass 2: fill observation indices
    for obs in 0..n_obs:
        for q in 0..Q:
            dof = offset[q] + level(obs, q)
            indices[pos[dof]] = obs
            pos[dof] += 1
```

**Residual update**: computes $r \leftarrow r - D^\top W (D \cdot \delta)$ restricted to observations touching the subdomain.

```
function ObservationSpaceUpdater::update(global_indices, correction, r_work):
    // 1. Map subdomain DOFs → positions
    for (pos, gi) in global_indices:
        dof_to_pos[gi] = pos           // sparse map, sentinel = u32::MAX

    // 2. Collect affected observations (dedup via visited bitset)
    touched_obs = []
    for gi in global_indices:
        for obs in obs_for_dof(gi):
            if not obs_visited[obs]:
                obs_visited[obs] = true
                touched_obs.push(obs)

    // 3. For each affected observation: accumulate and scatter
    for obs in touched_obs:
        t = 0
        for q in 0..Q:
            dof = offset[q] + level(obs, q)
            if dof_to_pos[dof] != SENTINEL:
                t += correction[dof_to_pos[dof]]

        if t != 0:
            wt = weight(obs) * t
            for q in 0..Q:
                r_work[offset[q] + level(obs, q)] -= wt

    // 4. Sparse cleanup: only reset touched entries
    for obs in touched_obs: obs_visited[obs] = false
    for gi in global_indices: dof_to_pos[gi] = SENTINEL
```

Cost: $O(|\text{affected\_obs}| \times Q)$ per subdomain, vs. $O(m)$ for a full operator application.

An alternative updater (`SparseGramianUpdater`) uses the explicit Gramian CSR directly: for each corrected DOF, scatter the Gramian row into the residual. Cost: $O(\text{nnz in touched rows})$.

---

## 8. Schur Complement Assembly

**Source**: `crates/within/src/operator/schur_complement.rs`

**Exact path** (`SchurLaplacian::from_elimination`): parallel row-workspace accumulation.

```
function from_elimination(elim):
    rows = (0..n_keep).par_iter().map_init(
        || (work: vec![0.0; n_keep], touched: vec![]),
        |(work, touched), i|:
            // Diagonal entry
            work[i] = diag_keep[i]
            touched.push(i)

            // Schur fill: S[i,j] -= (C_k2e[i,k] / D_elim[k]) * C_e2k[k,j]
            for k in keep_to_elim.row(i):
                scale = keep_to_elim[i,k] * inv_diag_elim[k]
                for j in elim_to_keep.row(k):
                    if work[j] == 0 and j != i:
                        touched.push(j)
                    work[j] -= scale * elim_to_keep[k,j]

            // Extract sparse row, reset workspace
            sort(touched)
            row = [(j, work[j]) for j in touched if work[j] != 0 or j == i]
            for j in touched: work[j] = 0
            touched.clear()
            return row
    ).collect()

    assemble_csr(rows, n_keep)
```

The `(work, touched)` pair is allocated once per Rayon task and reused across rows. Only touched entries are cleared, avoiding $O(n_{\text{keep}})$ cost per row.

**Approximate path** (`ApproxSchurComplement::compute`): parallel edge emission with merge-tree reduce.

```
function compute(cross_tab, config):
    elim = Elimination::new(cross_tab)
    emitter = SampledCliqueEmitter { seed, split }

    edges = (0..n_elim).par_iter()
        .fold(
            || (edges: vec![], scratch: SampledScratch::default()),
            |(edges, scratch), k|:
                star = elim.star(k)        // zero-copy neighborhood view
                if star.degree > 1:
                    // Copy neighbors to AoS buffer
                    buf = [(col, weight) for (col, weight) in star.neighbors]
                    seed_k = seed + k
                    if split <= 1:
                        clique_tree_sample(buf, star.diag, seed_k, edges)
                    else:
                        clique_tree_sample_multi(buf, split, seed_k, edges)
        )
        .map(|(edges, _)|:
            sort_and_dedup(edges)          // per-task sort + weight merge
            edges
        )
        .reduce(vec![], merge_dedup)       // merge-tree: sorted merge of sorted lists

    build_laplacian_csr(edges, n_keep)
```

The `sort_and_dedup` step sorts edges by `(lo_col, hi_col)` and merges duplicate edges by summing weights. The `merge_dedup` reduction is a standard sorted-merge of two sorted lists, yielding the global sorted edge list without a single $O(E \log E)$ sort.

The final CSR assembly (`build_laplacian_csr`) uses the sorted edge ordering to place lower-triangle, diagonal, and upper-triangle entries in correct column order per row without any per-row sorting.

---

## Source File Reference

| File | Content |
|------|---------|
| `crates/within/src/orchestrate.rs` | Public API: `solve()`, `solve_batch()` |
| `crates/within/src/solver.rs` | Krylov dispatch, preconditioner reuse |
| `crates/within/src/domain.rs` | `WeightedDesign`, gather/scatter |
| `crates/within/src/domain/factor_pairs.rs` | Subdomain construction, partition of unity |
| `crates/within/src/operator/gramian.rs` | Implicit `GramianOperator` |
| `crates/within/src/operator/gramian/explicit.rs` | CSR Gramian construction |
| `crates/within/src/operator/gramian/cross_tab.rs` | `CrossTab`, bipartite components |
| `crates/within/src/operator/local_solver.rs` | `ApproxCholSolver`, `BlockElimSolver` |
| `crates/within/src/operator/schur_complement.rs` | Exact/approximate Schur |
| `crates/within/src/operator/schwarz.rs` | Preconditioner builder |
| `crates/within/src/operator/preconditioner.rs` | Fused build path |
| `crates/within/src/operator/residual_update.rs` | Observation-space updater |
| `crates/schwarz-precond/src/schwarz.rs` | Additive/multiplicative Schwarz |
| `crates/schwarz-precond/src/domain.rs` | `SubdomainCore`, atomic ops |
| `crates/schwarz-precond/src/solve/cg.rs` | CG solver |
| `crates/schwarz-precond/src/solve/gmres.rs` | GMRES solver |
