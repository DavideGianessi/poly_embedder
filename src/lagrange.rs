use crate::field::{add, modular_inverse, mult, sub, Fe};

#[derive(Clone, Copy, Debug)]
pub struct Point {
    pub x: Fe,
    pub y: Fe,
}

pub fn generate_vanishing_polynomial(points: &Vec<Point>) -> Vec<Fe> {
    let mut px = vec![Fe::from(1u8)];
    for point in points {
        px = multiply_by_linear(&px, point.x);
    }
    px
}

pub fn lagrange_interpolate(points: &Vec<Point>, vanishing: &Vec<Fe>) -> Vec<Fe> {
    let n = points.len();
    if n == 0 {
        return vec![];
    }

    let px = vanishing;

    let mut result = vec![Fe::from(0u8); n];

    for i in 0..n {
        let mut denom = Fe::from(1u8);
        for j in 0..n {
            if i != j {
                let diff = sub(points[i].x, points[j].x);
                denom = mult(denom, diff);
            }
        }
        let weight = mult(points[i].y, modular_inverse(denom));
        let basis = divide_by_linear(px, points[i].x);
        for j in 0..basis.len() {
            let scaled = mult(basis[j], weight);
            result[j] = add(result[j], scaled);
        }
    }
    result
}

fn multiply_by_linear(poly: &[Fe], root: Fe) -> Vec<Fe> {
    let mut result = vec![Fe::from(0u8); poly.len() + 1];
    for i in 0..poly.len() {
        result[i] = sub(result[i], mult(poly[i], root));
    }
    for i in 0..poly.len() {
        result[i + 1] = add(result[i + 1], poly[i]);
    }
    result
}

fn divide_by_linear(poly: &[Fe], root: Fe) -> Vec<Fe> {
    let n = poly.len();
    if n < 2 {
        return vec![];
    }
    let mut quotient = vec![Fe::from(0u8); n - 1];
    let mut current = poly[n - 1];
    quotient[n - 2] = current;
    for i in (1..n - 1).rev() {
        current = add(poly[i], mult(current, root));
        quotient[i - 1] = current;
    }
    quotient
}
