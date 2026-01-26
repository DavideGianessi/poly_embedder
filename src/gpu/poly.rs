use rustacuda::memory::DeviceBuffer;
use rustacuda::prelude::*;
use crate::field::Fe;
use super::context::CudaContext;

pub fn gpu_sum_polynomials(
    ctx: &CudaContext,
    poly1: &[Fe],
    poly2: &[Fe],
) -> Result<Vec<Fe>, Box<dyn std::error::Error>> {
    let result_size = poly1.len().max(poly2.len());
    
    // Allocate and copy data to GPU
    let d_poly1 = DeviceBuffer::from_slice(poly1)?;
    let d_poly2 = DeviceBuffer::from_slice(poly2)?;
    let mut d_result = DeviceBuffer::new(result_size as u64)?;
    
    // Configure kernel launch
    let block_size = 256;
    let grid_size = (result_size + block_size - 1) / block_size;
    
    unsafe {
        // Get the kernel function from loaded module
        let kernel = ctx.module.get_function("sum_polynomials_flex_kernel")?;
        
        // Launch kernel with parameters
        launch!(kernel<<<grid_size, block_size, 0, ctx.stream>>>(
            d_poly1.as_device_ptr(),
            poly1.len() as i32,
            d_poly2.as_device_ptr(),
            poly2.len() as i32,
            d_result.as_device_ptr(),
            result_size as i32
        ))?;
    }
    
    // Wait for kernel to complete
    ctx.synchronize()?;
    
    // Copy result back to host
    let mut result = vec![Fe::from(0u8); result_size];
    d_result.copy_to(&mut result)?;
    
    Ok(result)
}

/// Helper: Create a GPU buffer from polynomial
pub fn to_gpu_buffer(poly: &[Fe]) -> Result<DeviceBuffer<Fe>, Box<dyn std::error::Error>> {
    DeviceBuffer::from_slice(poly)
}

/// Helper: Copy GPU buffer back to host
pub fn from_gpu_buffer(buffer: &DeviceBuffer<Fe>) -> Result<Vec<Fe>, Box<dyn std::error::Error>> {
    let mut result = vec![Fe::from(0u8); buffer.len()];
    buffer.copy_to(&mut result)?;
    Ok(result)
}
