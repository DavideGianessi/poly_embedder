mod fft;
mod field;
mod lagrange;
mod poly;
mod gpu;

use std::time::Instant;
use crate::field::Fe;
use crate::lagrange::{/*generate_vanishing_polynomial, lagrange_interpolate,*/ Point};
use crate::fft::fft_multiply;
use crate::poly::{generate_random_polynomial, sum_poly};
use crate::gpu::{CudaContext, gpu_generate_vanishing, gpu_lagrange_interpolate};
use rustacuda::memory::{DeviceBuffer,CopyDestination};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new()?;

    let file = File::open("input.txt")?;
    let mut reader = BufReader::new(file);

    let mut input = String::new();
    reader.read_line(&mut input)?;

    let mut iter = input.split_whitespace();
    let n_points: u32 = iter.next().unwrap().parse()?;
    let final_degree: u32 = iter.next().unwrap().parse()?;

    let mut points = Vec::new();
    let mut points_x = Vec::new();
    let mut points_y = Vec::new();

    for _ in 0..n_points {
        input.clear();
        reader.read_line(&mut input)?;
        let mut iter = input.split_whitespace();
        let x: Fe = iter.next().unwrap().parse()?;
        let y: Fe = iter.next().unwrap().parse()?;
        points.push(Point { x: x, y: y });
        points_x.push(x);
        points_y.push(y);
    }
    let start = Instant::now();
    let mut d_points_x = DeviceBuffer::from_slice(&points_x)?;
    let mut d_points_y = DeviceBuffer::from_slice(&points_y)?;
    println!("moving points to gpu memory: {:?}", start.elapsed());

    let start = Instant::now();
    let mut d_vanishing = gpu_generate_vanishing(&ctx, &mut d_points_x).expect("Failed to generate vanishing polynomial on GPU");
    println!("generate_vanishing_polynomial: {:?}", start.elapsed());


    let start = Instant::now();
    let points_per_thread = 8;
    let d_poly1 = gpu_lagrange_interpolate(&ctx,  &mut d_points_x, &mut d_points_y, &mut d_vanishing, n_points as usize, points_per_thread)?;
    let poly1_degree = points.len() - 1;
    println!("Lagrang interpolate: {:?}", start.elapsed());

    let vanishing = {
        ctx.synchronize()?;
        let mut v = vec![Fe::from(0u32); n_points as usize + 1];
        d_vanishing.copy_to(&mut v)?;
        v
    };
    let poly1 = {
        ctx.synchronize()?;
        let mut v = vec![Fe::from(0u32); n_points as usize];
        d_poly1.copy_to(&mut v)?;
        v
    };


    let start = Instant::now();
    let poly2 = generate_random_polynomial(final_degree - poly1_degree as u32 - 1);
    println!("Generate random polynomial: {:?}", start.elapsed());

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
