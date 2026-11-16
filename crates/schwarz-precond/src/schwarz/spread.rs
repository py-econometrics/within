//! An apply's budget spread over the subdomains by work: one past the budget's share on the whole budget, one after another, the rest at once, one thread each, when their work pays for waking the pool.

use dyn_stack::{MemStack, StackReq};

use crate::threads::Threads;

/// Work of the subdomains within the share below which they stay on the calling thread: approx-chol's break-even for waking the pool, in the same steps.
const SPREAD_WORK: usize = 8192;

/// Claims per worker: a few dozen let unequal subdomains even out.
pub(super) const CLAIMS_PER_WORKER: usize = 16;

/// What spreading an apply knows of one subdomain.
#[derive(Clone, Copy, Debug)]
pub(super) struct Cost {
    pub(super) work: usize,
    /// Its local solve's scratch on one thread.
    pub(super) scratch: StackReq,
}

/// How an apply at a budget runs the subdomains.
pub(super) struct Spread {
    /// Threads at a time: the budget's, at most the pool's and one per subdomain.
    tasks: usize,
    /// The budget's share of the work; a subdomain past it runs alone.
    share: usize,
    /// Summed work of the subdomains within the share.
    within: usize,
    /// The most scratch one subdomain within the share takes.
    pub(super) region: StackReq,
}

impl Spread {
    pub(super) fn of(costs: &[Cost], threads: Threads) -> Self {
        let tasks = threads.tasks().min(costs.len()).max(1);
        let total = costs
            .iter()
            .fold(0usize, |sum, c| sum.saturating_add(c.work));
        let share = total / tasks;
        let (mut within, mut region) = (0usize, StackReq::EMPTY);
        for cost in costs.iter().filter(|c| c.work <= share) {
            within = within.saturating_add(cost.work);
            region = region.or(cost.scratch);
        }
        Self {
            tasks,
            share,
            within,
            region,
        }
    }

    pub(super) fn alone(&self, cost: &Cost) -> bool {
        cost.work > self.share
    }

    /// Regions the subdomains within the share run on at once: one per task when they pay for the pool.
    pub(super) fn regions(&self) -> usize {
        match self.tasks > 1 && self.within >= SPREAD_WORK {
            true => self.tasks,
            false => 1,
        }
    }

    /// How many of the [`regions`](Self::regions) `stack` holds; at least one.
    pub(super) fn workers(&self, stack: &MemStack) -> usize {
        let tasks = self.regions();
        let fit = match self.region.size_bytes() {
            0 => tasks,
            size => (stack.len_bytes() / size).min(tasks),
        };
        (1..=fit)
            .rev()
            .find(|&count| stack.can_hold(self.region.array(count)))
            .unwrap_or(1)
    }
}

/// Runs `work` on `workers` regions of `region` carved from `stack`, each on a task of its own.
pub(super) fn split<E: Send>(
    stack: &mut MemStack,
    region: StackReq,
    workers: usize,
    work: &(impl Fn(&mut MemStack) -> Result<(), E> + Sync),
) -> Result<(), E> {
    if workers == 1 {
        return work(stack);
    }
    let half = workers / 2;
    let carved = region.array(half);
    let (mine, rest) = stack.make_aligned_uninit::<u8>(carved.size_bytes(), carved.align_bytes());
    let (left, right) = rayon::join(
        || split(MemStack::new(mine), region, half, work),
        || split(rest, region, workers - half, work),
    );
    left.and(right)
}
