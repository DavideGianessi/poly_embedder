mod fft;
mod field;
mod lagrange;
mod poly;
use std::time::Instant;

use crate::fft::fft_multiply;
use crate::field::Fe;
use crate::lagrange::{generate_vanishing_polynomial, lagrange_interpolate, Point};
use crate::poly::{generate_random_polynomial, sum_poly};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
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

    let start = Instant::now();
    let poly1 = lagrange_interpolate(&points);
    let poly1_degree = points.len() - 1;
    println!("Lagrang interpolate: {:?}", start.elapsed());


    let start = Instant::now();
    let poly2 = generate_random_polynomial(final_degree - poly1_degree as u32 - 1);
    println!("Generate random polynomial: {:?}", start.elapsed());

    let start = Instant::now();
    let vanishing = generate_vanishing_polynomial(&points);
    println!("generate_vanishing_polynomial: {:?}", start.elapsed());

    let start = Instant::now();
    let extended_poly = fft_multiply(poly2, vanishing);
    println!("fft multiply: {:?}", start.elapsed());

    let start = Instant::now();
    let final_poly = sum_poly(&extended_poly, &poly1);
    println!("final sum: {:?}", start.elapsed());

    let mut file = File::create("output.txt")?;

    writeln!(file, "{}", final_poly.len())?;

    for coeff in final_poly {
        writeln!(file, "{}", coeff)?;
    }
    println!("Done!");

    Ok(())
}
