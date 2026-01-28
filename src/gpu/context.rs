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
        rustacuda::init(CudaFlags::empty())?;
        
        let device = Device::get_device(0)?;
        
        let context = Context::create_and_push(
            ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO,
            device
        )?;
        
        let ptx_content = include_str!(env!("KERNELS_PTX"));
        let ptx = CString::new(ptx_content)?;
        let module = Module::load_from_string(&ptx)?;
        
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
