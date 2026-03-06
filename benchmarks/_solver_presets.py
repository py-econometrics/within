"""Shared benchmark solver presets aligned with the current Python API."""

from __future__ import annotations

from within import (
    ApproxCholConfig,
    ApproxSchurConfig,
    AdditiveSchwarz,
    CG,
    GMRES,
    MultiplicativeSchwarz,
    OperatorRepr,
    SchurComplement,
)

from ._registry import SuiteOptions
from ._types import SolverConfig


def schur_local_solver(seed: int, *, split: int = 1) -> SchurComplement:
    return SchurComplement(
        approx_chol=ApproxCholConfig(seed=seed, split=split),
        approx_schur=ApproxSchurConfig(seed=seed),
    )


def additive_schwarz(seed: int, *, split: int = 1) -> AdditiveSchwarz:
    return AdditiveSchwarz(local_solver=schur_local_solver(seed, split=split))


def multiplicative_schwarz(seed: int, *, split: int = 1) -> MultiplicativeSchwarz:
    return MultiplicativeSchwarz(local_solver=schur_local_solver(seed, split=split))


def cg_solver_config(
    opts: SuiteOptions,
    label: str = "CG(Schwarz)",
    *,
    split: int = 1,
    operator: OperatorRepr = OperatorRepr.Implicit,
    maxiter: int | None = None,
    preconditioned: bool = True,
) -> SolverConfig:
    return SolverConfig(
        label,
        CG(
            tol=opts.tol,
            maxiter=opts.maxiter if maxiter is None else maxiter,
            preconditioner=(
                additive_schwarz(opts.seed, split=split) if preconditioned else None
            ),
            operator=operator,
        ),
    )


def gmres_solver_config(
    opts: SuiteOptions,
    label: str = "GMRES(Mult-Schwarz)",
    *,
    split: int = 1,
    operator: OperatorRepr = OperatorRepr.Implicit,
    maxiter: int | None = None,
    multiplicative: bool = True,
) -> SolverConfig:
    return SolverConfig(
        label,
        GMRES(
            tol=opts.tol,
            maxiter=opts.maxiter if maxiter is None else maxiter,
            preconditioner=(
                multiplicative_schwarz(opts.seed, split=split)
                if multiplicative
                else additive_schwarz(opts.seed, split=split)
            ),
            operator=operator,
        ),
    )


def standard_solver_configs(
    opts: SuiteOptions,
    *,
    include_cg_none: bool = False,
    include_gmres_multiplicative: bool = True,
    maxiter: int | None = None,
) -> list[SolverConfig]:
    configs: list[SolverConfig] = []
    if include_cg_none:
        configs.append(
            cg_solver_config(
                opts,
                label="CG(none)",
                maxiter=maxiter,
                preconditioned=False,
            )
        )
    configs.append(cg_solver_config(opts, maxiter=maxiter))
    if include_gmres_multiplicative:
        configs.append(gmres_solver_config(opts, maxiter=maxiter))
    return configs
