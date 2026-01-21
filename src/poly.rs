use crate::field::{add, generate_random_element, Fe};


pub fn sum_poly(poly1: &Vec<Fe>, poly2: &Vec<Fe>) -> Vec<Fe> {
    let final_length = poly1.len().max(poly2.len());
    let mut res = vec![Fe::from(0u8); final_length];
    for i in 0..poly1.len() {
        res[i] = poly1[i];
    }
    for i in 0..poly2.len() {
        res[i] = add(res[i], poly2[i]);
    }
    res
}

pub fn generate_random_polynomial(degree: u32) -> Vec<Fe> {
    let mut poly = Vec::new();
    for _ in 0..(degree + 1) {
        poly.push(generate_random_element());
    }
    poly
}
