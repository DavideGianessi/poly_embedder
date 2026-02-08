# Poly Embedder

This is my project for the course "Accelerated elaboration systems" at the University of Bologna.

The program takes as input a file input.txt with the number of points `n`, the desired degree `d`, and `n` points themselves. It embeds those point into a polynomial of degree `d`.

The project features an initial sequential implementation, followed by multiple iterations where it gets progressively optimized for GPU.

## Project Structure

You can navigate the different optimization stages by switching branches. In order, the branches are:

- `sequential`: Baseline Rust implementation.
- `gpu_first_iteration`: Naive CUDA porting.
- `gpu_ntt_opt`: Optimized Number Theoretic Transform.
- `bit_reverse_opt`: Improved bit-reversal using **Shared Memory** and bank confict avoidance.
- `main`: Final version featuring a **Systolic Pipeline** to deal with Lagrange contributions.

## Performance Highlights

The following benchmarks were taken with $n = 30,000$ and $d = 10,000,000$ on an **AMD Ryzen 7 4800H** (CPU) and an **NVIDIA GeForce GTX 1650 Ti Mobile** (GPU).

| Stage | `Sequential` | `gpu_first_iteration` | `gpu_ntt_opt` | `bit_reverse_opt` | `main` |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Vanishing Polynomial** | 1.465 s | 8.9 ms | 8.9 ms | 8.9 ms | 9.1 ms |
| **Lagrange Interpolation** | 10.755 s | 473 ms | 480 ms | 475 ms | **48.7 ms** |
| **Generating random polynomial** | 100.6 ms | 5.62 ms | 5.61 ms | 5.59 ms | 7.9 ms |
| **NTT Multiplication** | 4.112 s | 240 ms | **124 ms** | **64.8 ms** | 70.8 ms |
| **Final sum** | 7.114 ms | 1.07 ms | 1.08 ms | 1.08 ms | 0.97 ms |

## Requirements

To build and run this project, you need:
- `cargo` (Rust)
- `nvcc` (CUDA Toolkit)
- `python` (for dataset generation and verification)

## Build Instructions

**Note:** Running `cargo clean` is required if you have already built the project and then switched branches. Using the `--release` flag is highly suggested, as it makes the sequential solution substantially faster.

```bash
cargo clean
cargo build --release
```

## Usage

### 1. Create a Dataset

Generate an input file with `n` points and degree `d`. The maximum value of `d` supported is 2^24.
```bash
python gen_input.py {n} {d}
```

### 2. Run the Program
```bash
./target/release/poly_embedder
```

### 3. Verify the Result
Run the verification script to ensure the generated polynomial correctly embeds the input points:
```bash
python verify.py
```

