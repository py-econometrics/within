//! Norms as `m · 2^e`, for comparing gradients of problems whose scales share no range.

use std::cmp::Ordering;
use std::ops::{Div, Mul};

/// A nonnegative `m · 2^e`, `m ∈ [1, 2)`, zero or non-finite, for expressions a few factors deep.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Magnitude {
    m: f64,
    e: i32,
}

impl Magnitude {
    pub(super) const ZERO: Self = Self { m: 0.0, e: 0 };

    /// `a · b`, exact in exponent where the `f64` product would over- or underflow.
    pub(super) fn product(a: f64, b: f64) -> Self {
        Self::from(a) * Self::from(b)
    }

    /// The nearest `f64`, saturating to 0 or ∞; a subnormal result rounds a second time.
    pub(super) fn to_f64(self) -> f64 {
        if !self.m.is_normal() || self.e >= f64::MAX_EXP {
            return if self.m.is_normal() {
                f64::INFINITY
            } else {
                self.m
            };
        }
        if self.e >= f64::MIN_EXP - 1 {
            return self.m * pow2(self.e);
        }
        // A subnormal result: the first factor is exact, so only the second rounds.
        self.m * pow2((self.e + 64).max(f64::MIN_EXP - 1)) * pow2(-64)
    }

    /// Neither zero, infinite nor NaN.
    pub(super) fn is_normal(self) -> bool {
        self.m.is_normal()
    }

    fn normalized(m: f64, e: i32) -> Self {
        if !m.is_normal() {
            return Self { m, e: 0 };
        }
        let Self { m, e: shift } = Self::from(m);
        Self { m, e: e + shift }
    }
}

impl From<f64> for Magnitude {
    fn from(x: f64) -> Self {
        if !x.is_finite() || x == 0.0 {
            return Self { m: x, e: 0 };
        }
        // A subnormal carries its leading bit below the mantissa field; 2^64 lifts it exactly.
        let (x, bias) = if x.is_normal() {
            (x, 0)
        } else {
            (x * pow2(64), -64)
        };
        debug_assert!(x > 0.0, "a norm, got {x:e}");
        let bits = x.to_bits();
        let e = ((bits >> 52) & 0x7ff) as i32 - 1023;
        let m = f64::from_bits((bits & ((1 << 52) - 1)) | (1023 << 52));
        Self { m, e: e + bias }
    }
}

impl Mul for Magnitude {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::normalized(self.m * rhs.m, self.e + rhs.e)
    }
}

impl Div for Magnitude {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Self::normalized(self.m / rhs.m, self.e - rhs.e)
    }
}

impl PartialOrd for Magnitude {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.m.is_normal() && other.m.is_normal() {
            return Some(self.e.cmp(&other.e).then(self.m.total_cmp(&other.m)));
        }
        // 0, ∞ and NaN order against any `m ∈ [1, 2)` as they do against its value.
        self.m.partial_cmp(&other.m)
    }
}

/// `2^e` for a normal exponent, built from its bits since `powi` underflows on the way there.
fn pow2(e: i32) -> f64 {
    debug_assert!((f64::MIN_EXP - 1..f64::MAX_EXP).contains(&e));
    f64::from_bits(((e + 1023) as u64) << 52)
}

#[cfg(test)]
mod tests {
    use super::Magnitude;
    use rstest::rstest;

    #[rstest]
    #[case::one(1.0)]
    #[case::large(3.7e300)]
    #[case::smallest_normal(f64::MIN_POSITIVE)]
    #[case::subnormal(1.3e-310)]
    #[case::smallest_subnormal(f64::from_bits(1))]
    #[case::largest(f64::MAX)]
    #[case::zero(0.0)]
    #[case::infinity(f64::INFINITY)]
    fn a_float_round_trips(#[case] x: f64) {
        assert_eq!(Magnitude::from(x).to_f64().to_bits(), x.to_bits());
    }

    /// Where the `f64` arithmetic stays in range the two agree bitwise.
    #[rstest]
    #[case::unit(3.0, 7.0, 5.0)]
    #[case::large(3e150, 7e150, 5e-7)]
    #[case::small(3e-150, 7e-150, 5e7)]
    fn in_range_arithmetic_is_the_float_arithmetic(#[case] a: f64, #[case] b: f64, #[case] c: f64) {
        let exact = Magnitude::product(a, b) / Magnitude::from(c);
        assert_eq!(exact.to_f64().to_bits(), (a * b / c).to_bits());
    }

    /// A quotient back in range survives a product that is not.
    #[rstest]
    #[case::overflowing_product(1e300, 3e300, 1e300, 3e300)]
    #[case::underflowing_product(1e-300, 3e-300, 1e-300, 3e-300)]
    #[case::subnormal_quotient(1e-300, 3e-10, 1.0, 3e-310)]
    fn out_of_range_products_divide_back(
        #[case] a: f64,
        #[case] b: f64,
        #[case] c: f64,
        #[case] expected: f64,
    ) {
        let got = (Magnitude::product(a, b) / Magnitude::from(c)).to_f64();
        assert!(
            (got / expected - 1.0).abs() < 1e-9,
            "{got:e} vs {expected:e}"
        );
    }

    #[test]
    fn the_order_is_the_value_order() {
        let tiny = Magnitude::product(1e-300, 1e-300);
        let huge = Magnitude::product(1e300, 1e300);
        assert!(Magnitude::ZERO < tiny && tiny < Magnitude::from(1.0));
        assert!(Magnitude::from(1.0) < huge && huge < Magnitude::from(f64::INFINITY));
        assert!(Magnitude::from(1.5) > Magnitude::from(1.25));
        assert_eq!(tiny.to_f64(), 0.0);
        assert_eq!(huge.to_f64(), f64::INFINITY);
        assert!(Magnitude::from(f64::NAN)
            .partial_cmp(&Magnitude::ZERO)
            .is_none());
    }
}
