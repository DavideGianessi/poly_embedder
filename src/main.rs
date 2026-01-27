//mod fft;
mod field;
mod lagrange;
//mod poly;
mod gpu;

use std::time::Instant;
use crate::field::Fe;
use crate::lagrange::{Point};
//use crate::poly::{generate_random_polynomial};
use crate::gpu::{CudaContext, gpu_generate_vanishing, gpu_lagrange_interpolate, gpu_fft_multiply, gpu_sum_polynomials, gpu_generate_random_polynomial};
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
    let mut d_poly1 = gpu_lagrange_interpolate(&ctx,  &mut d_points_x, &mut d_points_y, &mut d_vanishing, n_points as usize, points_per_thread)?;
    let poly1_degree = points.len() - 1;
    println!("Lagrang interpolate: {:?}", start.elapsed());

    let start = Instant::now();
    let mut d_poly2 = gpu_generate_random_polynomial(&ctx, (final_degree - poly1_degree as u32 - 1) as usize).expect("random poly generation failed");
    println!("Generate random polynomial: {:?}", start.elapsed());

    let start = Instant::now();
    let mut d_extended_poly = gpu_fft_multiply(&ctx, &mut d_poly2, &mut d_vanishing).expect("gpu fft failed");
    println!("fft multiply: {:?}", start.elapsed());

    let start = Instant::now();
    let d_final_poly = gpu_sum_polynomials(&ctx, &mut d_extended_poly, &mut d_poly1).expect("final sum failed");
    println!("final sum: {:?}", start.elapsed());

    let start = Instant::now();
    let final_poly = {
        ctx.synchronize()?;
        let mut v = vec![Fe::from(0u32); (final_degree +1) as usize];
        d_final_poly.copy_to(&mut v)?;
        v
    };
    println!("moving result back to cpu memory: {:?}", start.elapsed());

    let mut file = File::create("output.txt")?;

    writeln!(file, "{}", final_poly.len())?;

    for coeff in final_poly {
        writeln!(file, "{}", coeff)?;
    }
    println!("Done!");

    Ok(())
}
