#![allow(unused)]
use crate::field::{add, mult, sub, Fe, GENS, IGENS, N_INVS};

fn ntt(poly: &mut Vec<Fe>, order: u32, gens: &[Fe; 25]) {
    let n = 1 << order;
    assert!(poly.len() <= n, "poly too large for the chosen order");
    if poly.len() < n {
        poly.resize(n, Fe::from(0u8));
    }
    let root = gens[order as usize];

    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            poly.swap(i, j);
        }
    }

    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let mut wlen = root;
        for _ in 0..(n.trailing_zeros() - len.trailing_zeros()) {
            wlen = mult(wlen, wlen);
        }

        for i in (0..n).step_by(len) {
            let mut w = Fe::from(1u8);
            for j in 0..half {
                let idx1 = i + j;
                let idx2 = i + j + half;
                let u = poly[idx1];
                let v = mult(poly[idx2], w);
                poly[idx1] = add(u, v);
                poly[idx2] = sub(u, v);
                w = mult(w, wlen);
            }
        }
        len <<= 1;
    }
}

fn intt(poly: &mut Vec<Fe>, order: u32, igens: &[Fe; 25], n_invs: &[Fe; 25]) {
    let n = 1 << order;
    assert!(poly.len() == n, "poly is not of length 2^order");
    ntt(poly, order, igens);
    let n_inv = n_invs[order as usize];
    for coeff in poly.iter_mut() {
        *coeff = mult(*coeff, n_inv);
    }
}

pub fn fft_multiply(poly1: Vec<Fe>, poly2: Vec<Fe>) -> Vec<Fe> {
    let result_len = poly1.len() + poly2.len() - 1;
    let order = (result_len as f64).log2().ceil() as u32;
    let n = 1 << order;
    let mut a = poly1;
    a.resize(n, Fe::from(0u8));
    let mut b = poly2;
    b.resize(n, Fe::from(0u8));
    ntt(&mut a, order, &GENS);
    ntt(&mut b, order, &GENS);
    for i in 0..n {
        a[i] = mult(a[i], b[i]);
    }
    intt(&mut a, order, &IGENS, &N_INVS);
    a.truncate(result_len);
    a
}
