//! Compatibility symbols missing from the MKL package used by Candle.

use half::f16;
use std::ffi::c_char;

#[inline]
unsafe fn matrix_value(
    matrix: *const f16,
    transposed: bool,
    row: usize,
    col: usize,
    leading_dim: usize,
) -> f32 {
    let index = if transposed {
        col + row * leading_dim
    } else {
        row + col * leading_dim
    };
    (*matrix.add(index)).to_f32()
}

/// Candle 0.10 references the Fortran `hgemm_` symbol when MKL is enabled, but
/// the bundled MKL 2020 libraries do not export it. Keep F16 CPU matmul correct
/// with a scalar fallback; normal F32 inference continues to use MKL SGEMM.
#[no_mangle]
pub unsafe extern "C" fn hgemm_(
    transa: *const c_char,
    transb: *const c_char,
    m: *const i32,
    n: *const i32,
    k: *const i32,
    alpha: *const f16,
    a: *const f16,
    lda: *const i32,
    b: *const f16,
    ldb: *const i32,
    beta: *const f16,
    c: *mut f16,
    ldc: *const i32,
) {
    let (m, n, k) = (*m as usize, *n as usize, *k as usize);
    let (lda, ldb, ldc) = (*lda as usize, *ldb as usize, *ldc as usize);
    let transa = (*transa as u8).eq_ignore_ascii_case(&b't');
    let transb = (*transb as u8).eq_ignore_ascii_case(&b't');
    let alpha = (*alpha).to_f32();
    let beta = (*beta).to_f32();

    for col in 0..n {
        for row in 0..m {
            let mut sum = 0.0f32;
            for inner in 0..k {
                sum += matrix_value(a, transa, row, inner, lda)
                    * matrix_value(b, transb, inner, col, ldb);
            }
            let output = c.add(row + col * ldc);
            *output = f16::from_f32(alpha * sum + beta * (*output).to_f32());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hgemm_multiplies_column_major_matrices() {
        // [[1, 2], [3, 4]] * [[5, 6], [7, 8]] in column-major storage.
        let a = [1.0, 3.0, 2.0, 4.0].map(f16::from_f32);
        let b = [5.0, 7.0, 6.0, 8.0].map(f16::from_f32);
        let mut c = [f16::ZERO; 4];
        let (m, n, k, ld) = (2, 2, 2, 2);
        let (alpha, beta) = (f16::ONE, f16::ZERO);

        unsafe {
            hgemm_(
                b"N".as_ptr().cast(),
                b"N".as_ptr().cast(),
                &m,
                &n,
                &k,
                &alpha,
                a.as_ptr(),
                &ld,
                b.as_ptr(),
                &ld,
                &beta,
                c.as_mut_ptr(),
                &ld,
            );
        }

        assert_eq!(c.map(f16::to_f32), [19.0, 43.0, 22.0, 50.0]);
    }

    #[test]
    fn hgemm_supports_transposed_inputs_and_beta() {
        let a = [1.0, 2.0, 3.0, 4.0].map(f16::from_f32);
        let identity = [1.0, 0.0, 0.0, 1.0].map(f16::from_f32);
        let mut c = [f16::ONE; 4];
        let (m, n, k, ld) = (2, 2, 2, 2);
        let (alpha, beta) = (f16::ONE, f16::ONE);

        unsafe {
            hgemm_(
                b"T".as_ptr().cast(),
                b"N".as_ptr().cast(),
                &m,
                &n,
                &k,
                &alpha,
                a.as_ptr(),
                &ld,
                identity.as_ptr(),
                &ld,
                &beta,
                c.as_mut_ptr(),
                &ld,
            );
        }

        assert_eq!(c.map(f16::to_f32), [2.0, 4.0, 3.0, 5.0]);
    }
}
