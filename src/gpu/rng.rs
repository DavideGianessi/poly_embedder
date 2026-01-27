use rustacuda::launch;
use rustacuda::memory::DeviceBuffer;
use std::ffi::CStr;
use rand::RngCore;
use crate::gpu::context::CudaContext;
use crate::field::{Fe,P};

pub fn gpu_generate_random_polynomial(
    ctx: &CudaContext,
    degree: usize,
) -> Result<DeviceBuffer<Fe>, Box<dyn std::error::Error>> {
    let n = degree + 1;
    let stream = &ctx.stream;
    let mut d_poly = unsafe { DeviceBuffer::uninitialized(n)? };
    
    let mut seed = [0u32; 8];
    rand::thread_rng().fill_bytes(unsafe {
        std::slice::from_raw_parts_mut(seed.as_mut_ptr() as *mut u8, 32)
    });

    let kernel = ctx.module.get_function(CStr::from_bytes_with_nul(b"chacha20\0")?)?;
    
    let block_size = 256u32;
    let grid_size = (n as u32 + block_size - 1) / block_size;

    unsafe {
        launch!(kernel<<<grid_size, block_size, 0, stream>>>(
            d_poly.as_device_ptr(),
            n as u32,
            seed[0], seed[1], seed[2], seed[3],
            seed[4], seed[5], seed[6], seed[7],
            P
        ))?;
    }

    Ok(d_poly)
}
