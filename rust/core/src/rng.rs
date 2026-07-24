//! Port of the RNG portion of `evoc/common_nndescent.py` (lines 1-55): a
//! fast tausworthe-style PRNG shared by every stochastic step in evoc.
//!
//! State is `[i64; 3]` (matches the numba `i8[:]` shape-(3,) array). All
//! arithmetic in `tau_rand_int` is performed on wrapping i64 exactly as
//! numba/LLVM would do it (Rust's `i64` arithmetic operators already wrap
//! silently in release builds and panic-check in debug builds only via
//! explicit overflow, so we use `wrapping_*` throughout to match numba's
//! unchecked semantics in both profiles).

/// RNG state: three lanes of i64, matching the numba `i8[:]` (shape (3,)).
pub type RngState = [i64; 3];

/// Port of `seed`: fills all three lanes with `s + 0xFFFF`.
pub fn seed(rng_state: &mut RngState, s: i64) {
    let value = s.wrapping_add(0xFFFF);
    rng_state.fill(value);
}

/// Port of `tau_rand_int`. Returns a pseudo-random i32 (numba `i4` return
/// truncates the i64 XOR result via `as i32`, which is a two's-complement
/// truncation -- this is where negative values in the fixture come from).
pub fn tau_rand_int(state: &mut RngState) -> i32 {
    state[0] = (((state[0] & 4294967294) << 12) & 0xFFFFFFFF)
        ^ ((((state[0] << 13) & 0xFFFFFFFF) ^ state[0]) >> 19);
    state[1] = (((state[1] & 4294967288) << 4) & 0xFFFFFFFF)
        ^ ((((state[1] << 2) & 0xFFFFFFFF) ^ state[1]) >> 25);
    state[2] = (((state[2] & 4294967280) << 17) & 0xFFFFFFFF)
        ^ ((((state[2] << 3) & 0xFFFFFFFF) ^ state[2]) >> 11);

    (state[0] ^ state[1] ^ state[2]) as i32
}

/// Port of `tau_rand`: a pseudo-random f32 in `[0, 1]`.
///
/// PORT NOTE: numba's `float(integer)` converts the i32 to f64 (numba's
/// `float` is a 64-bit double), so the division `float(integer) / 0x7FFFFFFF`
/// happens in f64. `abs` is likewise applied in f64. Only the final return
/// value is narrowed to f32 by the `f4(...)` signature. We mirror that:
/// compute in f64, cast to f32 only at the end.
pub fn tau_rand(state: &mut RngState) -> f32 {
    let integer = tau_rand_int(state);
    (f64::from(integer) / 0x7FFFFFFFi64 as f64).abs() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_fills_all_three_lanes() {
        let mut state: RngState = [0; 3];
        seed(&mut state, 0);
        assert_eq!(state, [0xFFFF, 0xFFFF, 0xFFFF]);
    }

    #[test]
    fn seed_wraps_like_numba_i8_add() {
        let mut state: RngState = [0; 3];
        seed(&mut state, i64::MAX);
        // i64::MAX + 0xFFFF wraps around.
        assert_eq!(state[0], i64::MAX.wrapping_add(0xFFFF));
        assert_eq!(state[0], state[1]);
        assert_eq!(state[1], state[2]);
    }

    #[test]
    fn tau_rand_int_advances_state() {
        let mut state: RngState = [0xFFFF, 0xFFFF, 0xFFFF];
        let first = tau_rand_int(&mut state);
        let second = tau_rand_int(&mut state);
        assert_ne!(first, second);
    }

    #[test]
    fn tau_rand_is_bounded_and_advances_same_state() {
        let mut state: RngState = [0xFFFF, 0xFFFF, 0xFFFF];
        for _ in 0..100 {
            let v = tau_rand(&mut state);
            assert!((0.0..=1.0).contains(&v), "tau_rand out of bounds: {v}");
        }
    }

    #[test]
    fn tau_rand_int_zero_seed_first_value_matches_reference() {
        // Cross-checked against the Python reference by hand (see also the
        // fixture parity test in tests/rng.rs).
        let mut state: RngState = [0xFFFF, 0xFFFF, 0xFFFF];
        assert_eq!(tau_rand_int(&mut state), -267379553);
    }
}
