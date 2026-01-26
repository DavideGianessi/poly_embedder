use rustacuda::prelude::*;
use std::error::Error;
use std::ffi::CString;

pub struct CudaContext {
    pub module: Module,
    pub stream: Stream,
    _context: Context,
}

impl CudaContext {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        // 1. Initialize the CUDA Driver API
        rustacuda::init(CudaFlags::empty())?;
        
        // 2. Get the primary GPU (index 0)
        let device = Device::get_device(0)?;
        
        // 3. Create a context and push it to the top of the stack
        let context = Context::create_and_push(
            ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO,
            device
        )?;
        
        // 4. Load the PTX (compiled CUDA code)
        // This assumes your build.rs sets the KERNELS_PTX environment variable
        let ptx_path = std::env::var("KERNELS_PTX")
            .expect("KERNELS_PTX env var not set. Ensure build.rs is correctly configured.");
        
        let ptx_content = std::fs::read_to_string(ptx_path)?;
        let ptx = CString::new(ptx_content)?;
        let module = Module::load_from_string(&ptx)?;
        
        // 5. Create a stream for asynchronous execution
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        
        Ok(Self {
            _context: context,
            module,
            stream,
        })
    }

    /// Convenience method to synchronize the stream
    pub fn synchronize(&self) -> Result<(), Box<dyn Error>> {
        self.stream.synchronize()?;
        Ok(())
    }
}
