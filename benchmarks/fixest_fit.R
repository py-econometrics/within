# R fixest worker for the varying-slopes benchmark (#65).
#
# Reads a panel CSV, fits the given feols formula (warmup + median-of-n timed
# fits via proc.time()), and prints RESULT_* markers for the Python runner.
#
#   Rscript fixest_fit.R <csv_path> "<formula>" [n_repeat]

suppressMessages(library(fixest))
setFixest_nthreads(0) # 0 = all physical cores

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
    stop("usage: fixest_fit.R <csv_path> <formula> [n_repeat]")
}
csv_path <- args[[1]]
fml <- stats::as.formula(args[[2]])
n_repeat <- if (length(args) >= 3) as.integer(args[[3]]) else 3L

read_df <- function(path) {
    if (requireNamespace("data.table", quietly = TRUE)) {
        as.data.frame(data.table::fread(path))
    } else {
        read.csv(path)
    }
}

fit <- function(df) {
    feols(fml, data = df, fixef.tol = 1e-8, fixef.iter = 10000, notes = FALSE)
}

df <- read_df(csv_path)

m <- fit(df) # warmup
ts <- numeric(n_repeat)
for (k in seq_len(n_repeat)) {
    t0 <- proc.time()[["elapsed"]]
    m <- fit(df)
    ts[k] <- proc.time()[["elapsed"]] - t0
}

# Untimed sanity: count recovered slope components (fixest names them
# `factor[[slope]]`), confirming the slope terms were actually estimated.
n_slope_components <- tryCatch(
    {
        fe <- fixef(m, notes = FALSE)
        length(grep("\\[\\[", names(fe)))
    },
    error = function(e) NA_integer_
)

cat(sprintf("RESULT_TIME %.6f\n", stats::median(ts)))
cat(sprintf("RESULT_ITERS %s\n", paste(m$iterations, collapse = "/")))
cat(sprintf("RESULT_NTHREADS %d\n", getFixest_nthreads()))
cat(sprintf("RESULT_FIXEF %s\n", n_slope_components))
