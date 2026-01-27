use rustacuda::launch;
use rustacuda::memory::DeviceBuffer;
use std::ffi::CStr;
use crate::gpu::context::CudaContext;
use crate::field::Fe;

pub fn gpu_lagrange_interpolate(
    ctx: &CudaContext,
    points_x: &mut DeviceBuffer<Fe>,
    points_y: &mut DeviceBuffer<Fe>,
    vanishing_poly: &mut DeviceBuffer<Fe>,
    n_points: usize,
    points_per_thread: usize,
) -> Result<DeviceBuffer<Fe>, Box<dyn std::error::Error>> {
    let stream = &ctx.stream;

    let mut weights = unsafe { DeviceBuffer::<Fe>::uninitialized(n_points)? };
    let weight_kernel = ctx.module.get_function(CStr::from_bytes_with_nul(b"compute_weights\0")?)?;
    
    let block_size = 256u32;
    let grid_size = (n_points as u32 + block_size - 1) / block_size;

    unsafe {
        launch!(weight_kernel<<<grid_size, block_size, 0, stream>>>(
            points_x.as_device_ptr(),
            points_y.as_device_ptr(),
            weights.as_device_ptr(),
            n_points as i32
        ))?;
    }

    let num_threads = (n_points + points_per_thread - 1) / points_per_thread;
    let mut workspaces = unsafe { DeviceBuffer::<Fe>::zeroed(num_threads * n_points)? };

    let lagrange_kernel = ctx.module.get_function(CStr::from_bytes_with_nul(b"lagrange_contribution_batched\0")?)?;
    
    let contrib_grid = (num_threads as u32 + block_size - 1) / block_size;

    unsafe {
        launch!(lagrange_kernel<<<contrib_grid, block_size, 0, stream>>>(
            vanishing_poly.as_device_ptr(),
            points_x.as_device_ptr(),
            weights.as_device_ptr(),
            workspaces.as_device_ptr(),
            n_points as i32,
            points_per_thread as i32,
            num_threads as i32
        ))?;
    }

    let mut d_result = unsafe { DeviceBuffer::zeroed(n_points)? };
    let sum_kernel = ctx.module.get_function(CStr::from_bytes_with_nul(b"sum_workspaces\0")?)?;
    
    let sum_grid = (n_points as u32 + block_size - 1) / block_size;

    unsafe {
        launch!(sum_kernel<<<sum_grid, block_size, 0, stream>>>(
            workspaces.as_device_ptr(),
            d_result.as_device_ptr(),
            n_points as i32,
            num_threads as i32
        ))?;
    }

    ctx.synchronize()?;

    Ok(d_result)
}
