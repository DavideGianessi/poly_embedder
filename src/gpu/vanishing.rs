use rustacuda::memory::DeviceBuffer;
use crate::field::Fe;
use super::context::CudaContext;
use rustacuda::launch;
use rustacuda::memory::CopyDestination;
use std::ffi::CStr;
use rustacuda::memory::DeviceSlice;

pub fn gpu_generate_vanishing(
    ctx: &CudaContext,
    d_points_x: &mut DeviceBuffer<Fe>,
) -> Result<DeviceBuffer<Fe>, Box<dyn std::error::Error>> {
    let n_original = d_points_x.len();
    let n_padded = n_original.next_power_of_two();
    let log_n = n_padded.trailing_zeros();


    unsafe {
        let mut d_buf_a = DeviceBuffer::<Fe>::uninitialized(2 * n_padded)?;
        let mut d_buf_b = DeviceBuffer::<Fe>::uninitialized(2 * n_padded)?;

        let block_size = 256u32;
        let stream = &ctx.stream;
        let init_name = CStr::from_bytes_with_nul(b"init_vanishing\0")?;
        let init_kernel = ctx.module.get_function(init_name)?;
        let init_grid = (((n_padded) as u32 + block_size - 1) / block_size) as u32;
        
        launch!(init_kernel<<<init_grid, block_size, 0, stream>>>(
            d_points_x.as_device_ptr(),
            d_buf_a.as_device_ptr(),
            n_original as i32
        ))?;

        let van_name = CStr::from_bytes_with_nul(b"generate_vanishing\0")?;
        let vanishing_kernel = ctx.module.get_function(van_name)?;
        
        let mut src_ptr = d_buf_a.as_device_ptr();
        let mut dst_ptr = d_buf_b.as_device_ptr();

        for level in 0..log_n {
            let total_threads = 2*n_padded; 
            let grid_size = ((total_threads as u32 + block_size - 1) / block_size) as u32;

            launch!(vanishing_kernel<<<grid_size, block_size, 0, stream>>>(
                src_ptr,
                dst_ptr,
                level as i32
            ))?;
            std::mem::swap(&mut src_ptr, &mut dst_ptr);
        }

        ctx.synchronize()?;

        let mut result = DeviceBuffer::<Fe>::uninitialized(n_original + 1)?;
        let result_view = DeviceSlice::from_raw_parts(src_ptr, n_original + 1);
        result.copy_from(result_view)?;

        Ok(result)
    }
}
