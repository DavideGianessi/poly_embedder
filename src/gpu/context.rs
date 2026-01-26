use rustacuda::prelude::*;
use rustacuda::memory::DeviceBuffer;
use std::error::Error;

pub struct CudaContext {
    _context: Context,
    module: Module,
    stream: Stream,
}

impl CudaContext {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        // Initialize CUDA
        rustacuda::init(CudaFlags::empty())?;
        
        // Get first device
        let device = Device::get_device(0)?;
        
        // Create context
        let context = Context::create_and_push(
            ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO,
            device
        )?;
        
        // Load PTX module
        let ptx_path = std::env::var("KERNELS_PTX")
            .expect("KERNELS_PTX env var not set. Did build.rs run?");
        
        let ptx = CString::new(std::fs::read_to_string(ptx_path)?)?;
        let module = Module::load_from_string(&ptx)?;
        
        // Create stream
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        
        Ok(Self {
            _context: context,
            module,
            stream,
        })
    }
    
    pub fn launch_kernel<F>(
        &self,
        function_name: &str,
        grid_dim: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        shared_mem_bytes: u32,
        args: &[&dyn DevicePointer],
    ) -> Result<(), Box<dyn Error>> {
        let kernel = self.module.get_function(function_name)?;
        
        unsafe {
            launch!(kernel<<<grid_dim, block_dim, shared_mem_bytes, self.stream>>>(
                args
            ))?;
        }
        
        Ok(())
    }
    
    pub fn synchronize(&self) -> Result<(), Box<dyn Error>> {
        self.stream.synchronize()?;
        Ok(())
    }
}
