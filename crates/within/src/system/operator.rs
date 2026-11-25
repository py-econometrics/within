//! `A_r = (I − P_e) A_k`: the system matrix LSMR iterates on. Its two products run hundreds of
//! times a solve, so they read `A`'s rows [split](super::blocks) at fixed widths where fused.

#[cfg(test)]
mod tests;

use schwarz_precond::{MemStack, Operator, StackReq};

use super::blocks::{with_split, KeptSide, KeptTerms, ReducedKernel, RowScale, SplitRows};
use super::term::TermRow;
use super::ReducedDesign;

/// `A_r = (I − P_e) A_k`: the system matrix LSMR iterates on.
pub(crate) struct SystemMatrix<'p> {
    design: &'p ReducedDesign<'p>,
    /// `A_k` at the shape it runs at.
    kept: KeptSide<'p>,
}

impl<'p> SystemMatrix<'p> {
    pub(crate) fn new(design: &'p ReducedDesign<'p>) -> Self {
        let kept = KeptSide::new(&design.split);
        Self { design, kept }
    }
}

/// The kept columns `e`'s span absorbs, where `diag(S)` is 0: `A_r`'s column there is rounding, so
/// `x_j` stays 0 and LSMR stays off it.
pub(super) struct Absorbed(Vec<usize>);

impl Absorbed {
    /// The columns where `diag_s` is 0.
    pub(super) fn of(diag_s: &[f64]) -> Self {
        Self(
            (diag_s.iter().enumerate())
                .filter_map(|(j, &d)| (d == 0.0).then_some(j))
                .collect(),
        )
    }

    /// `x_j = 0` on each absorbed column.
    fn mask(&self, x: &mut [f64]) {
        for &j in &self.0 {
            x[j] = 0.0;
        }
    }
}

/// `y = A_r x`: per level, its segment of `A_k x` with `P_l` removed.
struct Forward<'a> {
    x: &'a [f64],
    y: &'a mut [f64],
    stack: &'a mut MemStack,
}

impl ReducedKernel for Forward<'_> {
    #[inline(always)]
    fn run<T: KeptTerms + Copy, R: TermRow, W: RowScale>(self, rows: &SplitRows<'_, T, R, W>) {
        rows.map_projected(self.y, self.stack, self.x);
    }
}

/// `x = A_rᵀ v = A_kᵀ (I − P_e) v`, level by level. It projects too: unprojected, rounding drifts
/// `x` out of `range(A_rᵀ)`. Unrescued: LSMR keeps `‖v‖` within `1e±64`, and the check costs 7%.
struct Adjoint<'a> {
    absorbed: &'a Absorbed,
    v: &'a [f64],
    x: &'a mut [f64],
    stack: &'a mut MemStack,
}

impl ReducedKernel for Adjoint<'_> {
    #[inline(always)]
    fn run<T: KeptTerms + Copy, R: TermRow, W: RowScale>(self, rows: &SplitRows<'_, T, R, W>) {
        rows.sum_projected(self.x, self.stack, self.v);
        self.absorbed.mask(self.x);
    }
}

/// `A_rᵀA_r = S`.
impl Operator for SystemMatrix<'_> {
    fn nrows(&self) -> usize {
        self.design.n_obs()
    }

    fn ncols(&self) -> usize {
        self.design.unknowns().len()
    }

    /// Each task's fit, and the adjoint's leaf accumulators.
    fn scratch(&self) -> StackReq {
        self.design.split.dynamic().scratch(self.ncols())
    }

    fn apply_with(
        &self,
        x: &[f64],
        y: &mut [f64],
        stack: &mut MemStack,
    ) -> Result<(), schwarz_precond::SolveError> {
        debug_assert_eq!((x.len(), y.len()), (self.ncols(), self.nrows()));
        with_split(&self.design.split, &self.kept, Forward { x, y, stack });
        Ok(())
    }

    fn apply_adjoint_with(
        &self,
        v: &[f64],
        x: &mut [f64],
        stack: &mut MemStack,
    ) -> Result<(), schwarz_precond::SolveError> {
        debug_assert_eq!((v.len(), x.len()), (self.nrows(), self.ncols()));
        with_split(
            &self.design.split,
            &self.kept,
            Adjoint {
                absorbed: &self.design.absorbed,
                v,
                x,
                stack,
            },
        );
        Ok(())
    }
}
