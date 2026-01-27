use rustacuda::memory::DeviceBuffer;
use rustacuda::launch;
use crate::field::Fe;
use std::ffi::CStr;
use super::context::CudaContext;

pub fn gpu_sum_polynomials(
    ctx: &CudaContext,
    d_poly1: &mut DeviceBuffer<Fe>,
    d_poly2: &mut DeviceBuffer<Fe>,
) -> Result<DeviceBuffer<Fe>, Box<dyn std::error::Error>> {
    let stream = &ctx.stream;
    let size1 = d_poly1.len();
    let size2 = d_poly2.len();
    let result_size = size1.max(size2);
    
    let mut d_result = unsafe { DeviceBuffer::<Fe>::zeroed(result_size)? };
    
    let block_size = 256u32;
    let grid_size = (result_size as u32 + block_size - 1) / block_size;
    
    unsafe {
        let kernel = ctx.module.get_function(CStr::from_bytes_with_nul(b"sum_polynomials\0")?)?;
        
        launch!(kernel<<<grid_size, block_size, 0, stream>>>(
            d_poly1.as_device_ptr(),
            size1 as i32,
            d_poly2.as_device_ptr(),
            size2 as i32,
            d_result.as_device_ptr(),
            result_size as i32
        ))?;
    }
    
    ctx.synchronize()?;
    
    Ok(d_result)
}
