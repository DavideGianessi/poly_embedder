use rustacuda::launch;
use rustacuda::memory::{DeviceBuffer, CopyDestination};
use std::ffi::CStr;
use crate::gpu::context::CudaContext;
use crate::field::{Fe, GENS, IGENS, N_INVS};

pub fn gpu_fft_multiply(
    ctx: &CudaContext,
    poly1: &DeviceBuffer<Fe>,
    poly2: &DeviceBuffer<Fe>,
) -> Result<DeviceBuffer<Fe>, Box<dyn std::error::Error>> {
    let stream = &ctx.stream;

    let min_size = poly1.len() + poly2.len() - 1;
    let n = min_size.next_power_of_two();
    let log_n = n.trailing_zeros() as usize;

    let mut d_a = unsafe { DeviceBuffer::zeroed(n)? };
    let mut d_b = unsafe { DeviceBuffer::zeroed(n)? };
    
    d_a[0..poly1.len()].copy_from(poly1)?;
    d_b[0..poly2.len()].copy_from(poly2)?;

    let mut d_roots = precompute_twiddles(ctx, n, log_n, false)?;
    let mut d_iroots = precompute_twiddles(ctx, n, log_n, true)?;

    gpu_ntt(ctx, &mut d_a, &mut d_roots, n, log_n)?;
    gpu_ntt(ctx, &mut d_b, &mut d_roots, n, log_n)?;

    let mut d_res = unsafe { DeviceBuffer::uninitialized(n)? };
    let pw_kernel = ctx.module.get_function(CStr::from_bytes_with_nul(b"pointwise_multiplication\0")?)?;
    let block_size = 256u32;
    let grid_pw = (n as u32 + block_size - 1) / block_size;

    unsafe {
        launch!(pw_kernel<<<grid_pw, block_size, 0, stream>>>(
            d_a.as_device_ptr(),
            d_b.as_device_ptr(),
            n as i32,
            d_res.as_device_ptr()
        ))?;
    }

    gpu_ntt(ctx, &mut d_res, &mut d_iroots, n, log_n)?;

    let scale_kernel = ctx.module.get_function(CStr::from_bytes_with_nul(b"intt_scale\0")?)?;
    let n_inv = N_INVS[log_n];
    let grid_scale = (n as u32 + block_size - 1) / block_size;

    unsafe {
        launch!(scale_kernel<<<grid_scale, block_size, 0, stream>>>(
            d_res.as_device_ptr(),
            n as u32,
            n_inv
        ))?;
    }

    ctx.synchronize()?;
    let mut truncated_result = unsafe { DeviceBuffer::uninitialized(min_size)? };
    truncated_result.copy_from(&d_res[0..min_size])?;
    Ok(truncated_result)
}

fn gpu_ntt(
    ctx: &CudaContext,
    data: &mut DeviceBuffer<Fe>,
    twiddles: &mut DeviceBuffer<Fe>,
    n: usize,
    log_n: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let stream = &ctx.stream;
    let block_size = 256u32;
    let grid_size = (n as u32 + block_size - 1) / block_size;
    let half_grid = ((n / 2) as u32 + block_size - 1) / block_size;

    let bit_rev_kernel = ctx.module.get_function(CStr::from_bytes_with_nul(b"bit_reverse\0")?)?;
    unsafe {
        launch!(bit_rev_kernel<<<grid_size, block_size, 0, stream>>>(
            data.as_device_ptr(),
            n as u32,
            log_n as u32
        ))?;
    }

    let step_kernel = ctx.module.get_function(CStr::from_bytes_with_nul(b"ntt_step\0")?)?;
    let mut len = 2u32;
    while len <= n as u32 {
        let half = len / 2;
        unsafe {
            launch!(step_kernel<<<half_grid, block_size, 0, stream>>>(
                data.as_device_ptr(),
                twiddles.as_device_ptr(),
                n as u32,
                len,
                half
            ))?;
        }
        len <<= 1;
    }

    Ok(())
}

fn precompute_twiddles(
    ctx: &CudaContext,
    n: usize,
    log_n: usize,
    inverse: bool
) -> Result<DeviceBuffer<Fe>, Box<dyn std::error::Error>> {
    let stream = &ctx.stream;
    let mut d_roots = unsafe { DeviceBuffer::uninitialized(n / 2)? };
    let kernel = ctx.module.get_function(CStr::from_bytes_with_nul(b"compute_twiddles\0")?)?;
    
    let base_root = if inverse { IGENS[log_n] } else { GENS[log_n] };
    let block_size = 256u32;
    let grid_size = ((n / 2) as u32 + block_size - 1) / block_size;

    unsafe {
        launch!(kernel<<<grid_size, block_size, 0, stream>>>(
            d_roots.as_device_ptr(),
            n as u32,
            base_root
        ))?;
    }
    Ok(d_roots)
}
