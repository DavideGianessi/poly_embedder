mod fft;
mod field;
mod lagrange;
mod poly;

use crate::fft::fft_multiply;
use crate::field::Fe;
use crate::lagrange::{generate_vanishing_polynomial, lagrange_interpolate, Point};
use crate::poly::{generate_random_polynomial, sum_poly};
use std::io::{BufRead, BufReader, Write};
use std::fs::File;



fn main()-> Result<(), Box<dyn std::error::Error>> {
    let file = File::open("input.txt")?;
    let mut reader = BufReader::new(file);

    let mut input = String::new();
    reader.read_line(&mut input)?;

    let mut iter = input.split_whitespace();
    let n_points: u32 = iter.next().unwrap().parse()?;
    let final_degree: u32 = iter.next().unwrap().parse()?;

    let mut points = Vec::new();

    for _ in 0..n_points {
        input.clear();
        reader.read_line(&mut input)?;
        let mut iter = input.split_whitespace();
        let x: Fe = iter.next().unwrap().parse()?;
        let y: Fe = iter.next().unwrap().parse()?;
        points.push(Point { x: x, y: y });
    }

    let vanishing = generate_vanishing_polynomial(&points);

    let poly1 = lagrange_interpolate(&vanishing, &points);

    let poly1_degree = points.len() - 1;

    let poly2 = generate_random_polynomial(final_degree - poly1_degree as u32 - 1);

    let extended_poly = fft_multiply(poly2, vanishing);

    let final_poly = sum_poly(&extended_poly, &poly1);



    let mut file = File::create("output.txt")?;

    writeln!(file, "{}", final_poly.len())?;

    for coeff in final_poly {
        writeln!(file, "{}", coeff)?;
    }
    println!("Done!");

    Ok(())

}
