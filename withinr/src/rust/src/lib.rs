// Extendr bridge exposing the `within` Rust crate to R as the `withinr` package.
//
// The bridge mirrors the Python binding (`crates/within-py`) in both shape and
// module layout, so core API changes port mechanically between the two:
//
// - `api`     — solve entry points and the persistent solver handle
// - `config`  — R config objects → native `within` config conversions, plus
//               the built-preconditioner handle
// - `convert` — shared R ↔ Rust coercion helpers and error plumbing
// - `results` — native result → R list conversions

use extendr_api::prelude::*;

mod api;
mod config;
mod convert;
mod results;

extendr_module! {
    mod withinr;
    use api;
    use config;
}
