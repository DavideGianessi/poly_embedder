use rustacuda::launch;
use rustacuda::memory::{DeviceBuffer, CopyDestination};
use std::ffi::CStr;
use crate::gpu::context::CudaContext;
use crate::field::{Fe, N_INVS};

pub fn gpu_fft_multiply(
    ctx: &CudaContext,
    poly1: &DeviceBuffer<Fe>,
    rand_poly: &DeviceBuffer<Fe>,
) -> Result<DeviceBuffer<Fe>, Box<dyn std::error::Error>> {
    let stream = &ctx.stream;

    let min_size:usize = poly1.len() + rand_poly.len() - 1;
    let n:usize = min_size.next_power_of_two();
    let log_n:usize = n.trailing_zeros().try_into().unwrap();

    let mut d_a = unsafe { DeviceBuffer::zeroed(n)? };
    let mut d_b = unsafe { DeviceBuffer::zeroed(n)? };
    
    d_a[0..poly1.len()].copy_from(poly1)?;
    d_b[0..rand_poly.len()].copy_from(rand_poly)?;

    let twiddle_size:usize = n/2;
    let mut d_roots = precompute_twiddles(ctx, n, log_n, twiddle_size, false)?;
    let mut d_iroots = precompute_twiddles(ctx, n, log_n, twiddle_size, true)?;

    bit_reverse(ctx, &mut d_a, n, log_n)?;
    bit_reverse(ctx, &mut d_b, n, log_n)?;

    gpu_ntt(ctx, &mut d_a, &mut d_roots, n, log_n, twiddle_size)?;
    gpu_ntt(ctx, &mut d_b, &mut d_roots, n, log_n, twiddle_size)?;


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

    bit_reverse(ctx, &mut d_res, n, log_n)?;
    gpu_ntt(ctx, &mut d_res, &mut d_iroots, n, log_n, twiddle_size)?;

    let scale_kernel = ctx.module.get_function(CStr::from_bytes_with_nul(b"intt_scale\0")?)?;
    let n_inv: u32 = N_INVS[log_n];
    let grid_scale = (n as u32 + block_size - 1) / block_size;

    unsafe {
        launch!(scale_kernel<<<grid_scale, block_size, 0, stream>>>(
            d_res.as_device_ptr(),
            n as u32,
            n_inv
        ))?;
    }

    ctx.synchronize()?;
    let mut truncated_result = unsafe { DeviceBuffer::zeroed(min_size)? };
    truncated_result.copy_from(&d_res[0..min_size])?;
    Ok(truncated_result)
}
fn bit_reverse(
    ctx: &CudaContext,
    data: &mut DeviceBuffer<Fe>,
    n: usize,
    log_n: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let stream = &ctx.stream;
    let bit_rev_kernel = ctx.module.get_function(CStr::from_bytes_with_nul(b"bit_reverse\0")?)?;
    let block_size = 256u32;
    let grid_size = (n as u32 + block_size - 1) / block_size;
    unsafe {
        launch!(bit_rev_kernel<<<grid_size, block_size, 0, stream>>>(
            data.as_device_ptr(),
            n as u32,
            log_n as u32
        ))?;
    }
    Ok(())
}
fn precompute_twiddles(
    ctx: &CudaContext,
    n: usize,
    log_n: usize,
    twiddle_size: usize,
    inverse: bool
) -> Result<DeviceBuffer<Fe>, Box<dyn std::error::Error>> {
    let stream = &ctx.stream;
    let log_twiddle: usize = twiddle_size.trailing_zeros().try_into().unwrap();
    let mut d_roots = unsafe { DeviceBuffer::uninitialized(twiddle_size*log_n)? };
    let kernel = ctx.module.get_function(CStr::from_bytes_with_nul(b"compute_twiddles\0")?)?;
    
    let block_size = 256u32;
    let grid_size = ( n as u32 + block_size - 1) / block_size;

    unsafe {
        launch!(kernel<<<grid_size, block_size, 0, stream>>>(
            d_roots.as_device_ptr(),
            n as u32,
            log_n as u32,
            twiddle_size as u32,
            log_twiddle as u32,
            inverse as bool
        ))?;
    }
    Ok(d_roots)
}
fn gpu_ntt(
    ctx: &CudaContext,
    data: &mut DeviceBuffer<Fe>,
    twiddles: &mut DeviceBuffer<Fe>,
    n: usize,
    log_n: usize,
    twiddle_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let stream = &ctx.stream;

    let threshold:usize = if n>= 2048 {11} else {0};

    if threshold == 11 {
        let locality_kernel = ctx.module.get_function(CStr::from_bytes_with_nul(b"ntt_block\0")?)?;
        let work_per_block = 2048u32;
        let locality_block = 1024u32;
        let locality_grid = (n as u32) / work_per_block;
        let smem_bytes = work_per_block * 4;
        unsafe {
            launch!(locality_kernel<<<locality_grid, locality_block, smem_bytes, stream>>>(
                data.as_device_ptr(),
                twiddles.as_device_ptr(),
                n as u32,
                twiddle_size as u32
            ))?;
        }
    }

    let step_kernel = ctx.module.get_function(CStr::from_bytes_with_nul(b"ntt_step\0")?)?;
    let block_size = 256u32;
    let grid_size = ((n / 2) as u32 + block_size - 1) / block_size;
    
    for log_half in threshold..log_n {
        let half: usize = 1<<log_half;
        let log_len: usize = log_half + 1;
        let len: usize = 1<<log_len;
        let twiddle_offset: usize = twiddle_size*log_half;
        unsafe {
            launch!(step_kernel<<<grid_size, block_size, 0, stream>>>(
                data.as_device_ptr(),
                twiddles.as_device_ptr(),
                n as u32,
                log_len as u32,
                len as u32,
                log_half as u32,
                half as u32,
                twiddle_offset as u32
            ))?;
        }
    }

    Ok(())
}
