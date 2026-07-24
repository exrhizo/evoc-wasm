//! Distance/dissimilarity kernels ported from the numba reference:
//! `float_nndescent.py::fast_cosine`,
//! `int8_nndescent.py::fast_int_inner_product_dissimilarity`, and
//! `uint8_nndescent.py::fast_bit_jaccard` / `fast_bit_jaccard_u64`.
//!
//! PORTING.md numeric parity policy applies: these are plain sequential
//! loops with the accumulator width the reference's numba `locals=`
//! declares -- no manual vectorization, no widening/narrowing beyond what
//! the reference specifies, even though the numba build itself compiles
//! with `fastmath=True` (which may reorder Python's own results slightly;
//! Rust/Zig must still be bit-identical to each other by staying strictly
//! sequential).

/// Smallest positive normal f32, `np.finfo(np.float32).tiny`.
const EXP_NEG_INF: f32 = f32::MIN_POSITIVE;

/// Port of `fast_cosine`. Accumulator is f32 (numba `locals={"result":
/// float32, ...}`), plain sequential summation.
pub fn fast_cosine(x: &[f32], y: &[f32]) -> f32 {
    let mut result: f32 = 0.0;
    let dim = x.len();
    for i in 0..dim {
        result += x[i] * y[i];
    }

    if result > 0.0 {
        -result
    } else {
        -EXP_NEG_INF
    }
}

/// Port of `fast_int_inner_product_dissimilarity`. Accumulates
/// `i32(x[i]) * i32(y[i])` in i32 (wrapping, matching numba's unchecked
/// integer ops), then negates the i32→f32 cast.
pub fn fast_int_inner_product_dissimilarity(x: &[i8], y: &[i8]) -> f32 {
    let mut result: i32 = 0;
    let dim = x.len();
    for i in 0..dim {
        result = result.wrapping_add((x[i] as i32).wrapping_mul(y[i] as i32));
    }
    -(result as f32)
}

/// Hardware-popcount helper matching numba's `popcnt_u8` LLVM intrinsic
/// (`llvm.ctpop.i8`); `u8::count_ones` compiles to the same instruction.
#[inline]
fn popcnt_u8(val: u8) -> u32 {
    val.count_ones()
}

/// Hardware-popcount helper matching numba's `popcnt_u64` LLVM intrinsic
/// (`llvm.ctpop.i64`).
#[inline]
fn popcnt_u64(val: u64) -> u32 {
    val.count_ones()
}

/// Port of `fast_bit_jaccard`. Per-byte popcounts of `x & y` and `x | y`
/// accumulated in u32.
pub fn fast_bit_jaccard(x: &[u8], y: &[u8]) -> f32 {
    let mut result: u32 = 0;
    let mut denom: u32 = 0;
    let dim = x.len();

    for i in 0..dim {
        let and_val = x[i] & y[i];
        let or_val = x[i] | y[i];
        result += popcnt_u8(and_val);
        denom += popcnt_u8(or_val);
    }

    if denom > 0 {
        -((result as f32) / (denom as f32))
    } else {
        0.0
    }
}

/// Port of `fast_bit_jaccard_u64`. Reads the u8 arrays 8 bytes at a time as
/// little-endian u64 (the numba `load_u64_from_u8_array` intrinsic loads
/// native-endian, and evoc targets little-endian platforms; the fixture
/// data confirms LE interpretation), popcounts in u64 accumulators.
///
/// REQUIRES: `x.len()` (== `y.len()`) divisible by 8, matching the Python
/// docstring's "REQUIRES: Array size divisible by 8."
pub fn fast_bit_jaccard_u64(x: &[u8], y: &[u8]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    debug_assert_eq!(x.len() % 8, 0, "fast_bit_jaccard_u64 requires len divisible by 8");

    let mut result: u64 = 0;
    let mut denom: u64 = 0;

    let n_u64 = x.len() / 8;

    for i in 0..n_u64 {
        let offset = i * 8;

        let x_val = u64::from_le_bytes(x[offset..offset + 8].try_into().unwrap());
        let y_val = u64::from_le_bytes(y[offset..offset + 8].try_into().unwrap());

        let and_val = x_val & y_val;
        let or_val = x_val | y_val;

        result += popcnt_u64(and_val) as u64;
        denom += popcnt_u64(or_val) as u64;
    }

    if denom > 0 {
        -((result as f32) / (denom as f32))
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_cosine_zero_vectors_returns_neg_tiny() {
        let x = [0.0f32; 4];
        let y = [0.0f32; 4];
        assert_eq!(fast_cosine(&x, &y), -EXP_NEG_INF);
    }

    #[test]
    fn fast_cosine_identical_unit_vectors() {
        let x = [1.0f32, 0.0, 0.0];
        let y = [1.0f32, 0.0, 0.0];
        assert_eq!(fast_cosine(&x, &y), -1.0);
    }

    #[test]
    fn fast_cosine_negative_dot_returns_neg_tiny() {
        let x = [1.0f32, 0.0];
        let y = [-1.0f32, 0.0];
        assert_eq!(fast_cosine(&x, &y), -EXP_NEG_INF);
    }

    #[test]
    fn fast_cosine_single_element() {
        let x = [2.0f32];
        let y = [3.0f32];
        assert_eq!(fast_cosine(&x, &y), -6.0);
    }

    #[test]
    fn fast_cosine_empty_returns_neg_tiny() {
        let x: [f32; 0] = [];
        let y: [f32; 0] = [];
        assert_eq!(fast_cosine(&x, &y), -EXP_NEG_INF);
    }

    #[test]
    fn fast_int_inner_product_zero_vectors() {
        let x = [0i8; 4];
        let y = [0i8; 4];
        assert_eq!(fast_int_inner_product_dissimilarity(&x, &y), 0.0);
    }

    #[test]
    fn fast_int_inner_product_single_element() {
        let x = [5i8];
        let y = [7i8];
        assert_eq!(fast_int_inner_product_dissimilarity(&x, &y), -35.0);
    }

    #[test]
    fn fast_int_inner_product_negative_values() {
        let x = [-128i8, 127];
        let y = [-128i8, 127];
        // (-128*-128) + (127*127) = 16384 + 16129 = 32513
        assert_eq!(fast_int_inner_product_dissimilarity(&x, &y), -32513.0);
    }

    #[test]
    fn fast_bit_jaccard_identical_vectors_is_neg_one() {
        let x = [0b1010_1010u8, 0b0000_1111];
        assert_eq!(fast_bit_jaccard(&x, &x), -1.0);
    }

    #[test]
    fn fast_bit_jaccard_zero_vectors_returns_zero() {
        let x = [0u8; 4];
        let y = [0u8; 4];
        assert_eq!(fast_bit_jaccard(&x, &y), 0.0);
    }

    #[test]
    fn fast_bit_jaccard_disjoint_bits() {
        let x = [0b1111_0000u8];
        let y = [0b0000_1111u8];
        // and=0, or=8 bits -> result 0
        assert_eq!(fast_bit_jaccard(&x, &y), 0.0);
    }

    #[test]
    fn fast_bit_jaccard_u64_matches_byte_version_on_identical_vectors() {
        let x: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(fast_bit_jaccard_u64(&x, &x), -1.0);
        assert_eq!(fast_bit_jaccard(&x, &x), -1.0);
    }

    #[test]
    fn fast_bit_jaccard_u64_zero_vectors_returns_zero() {
        let x = [0u8; 16];
        let y = [0u8; 16];
        assert_eq!(fast_bit_jaccard_u64(&x, &y), 0.0);
    }

    #[test]
    fn fast_bit_jaccard_u64_single_chunk() {
        let x: [u8; 8] = [0xFF, 0, 0, 0, 0, 0, 0, 0];
        let y: [u8; 8] = [0x0F, 0, 0, 0, 0, 0, 0, 0];
        // and popcount = 4, or popcount = 8
        assert_eq!(fast_bit_jaccard_u64(&x, &y), -0.5);
    }
}
