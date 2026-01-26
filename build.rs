use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=cuda/kernels.cu");
    println!("cargo:rerun-if-changed=cuda/field.cuh");
    
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    
    let status = Command::new("nvcc")
        .args(&[
            "-ptx",
            "-O3",
            "-arch=sm_75",
            "-std=c++14",
            "--generate-line-info",
            "-Xcompiler", "-Wall",
            "-o",
            out_dir.join("kernels.ptx").to_str().unwrap(),
            "cuda/kernels.cu",
        ])
        .status()
        .expect("Failed to run nvcc. Make sure CUDA is installed: nvcc --version");
    
    if !status.success() {
        panic!("CUDA compilation failed");
    }
    
    println!("cargo:rustc-env=KERNELS_PTX={}", 
             out_dir.join("kernels.ptx").display());
}
