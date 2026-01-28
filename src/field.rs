use rand::RngCore;

pub type Fe = u32;
pub type DoubleFe = u64;
pub const FE_BITS: u32 = 32;
pub const P: Fe=4194304001;
pub const MAGIC: DoubleFe = 4398046510;
pub const GENS: [Fe; 25]= [1, 4194304000, 809539273, 2303415184, 1800537630, 2906399817, 369001549, 2026377158, 1867760616, 3185713831, 3100728574, 3986884701, 2037177755, 3682666484, 1581848693, 217320144, 623292090, 502725452, 790764273, 1079588648, 3440443607, 1688530187, 2541931790, 2936257672, 2580763344];
pub const IGENS: [Fe; 25]= [1, 4194304000, 3384764728, 3412379098, 1559634102, 1560690925, 1481810193, 3824470519, 306209204, 235196417, 402301397, 4159660757, 3602029040, 2380151834, 1885459, 2469224405, 3336134804, 3231469334, 1976201916, 4149395070, 1476203138, 1004409423, 3013869102, 2962262218, 3810123335];
pub const N_INVS: [Fe; 25]= [1, 2097152001, 3145728001, 3670016001, 3932160001, 4063232001, 4128768001, 4161536001, 4177920001, 4186112001, 4190208001, 4192256001, 4193280001, 4193792001, 4194048001, 4194176001, 4194240001, 4194272001, 4194288001, 4194296001, 4194300001, 4194302001, 4194303001, 4194303501, 4194303751];

pub fn mult(a: Fe, b: Fe) -> Fe {
    let x = a as DoubleFe * b as DoubleFe;
    let q = ((x >> FE_BITS) * MAGIC) >> FE_BITS;
    let r = x - q * (P as DoubleFe);
    let mut r = r as DoubleFe;
    if r >= P as DoubleFe {
        r -= P as DoubleFe;
    }
    if r >= P as DoubleFe {
        r -= P as DoubleFe;
    }
    r as Fe
}

pub fn add(a: Fe, b: Fe) -> Fe {
    let mut x = a as DoubleFe + b as DoubleFe;
    if x >= P as DoubleFe {
        x -= P as DoubleFe;
    }
    x as Fe
}

pub fn sub(a: Fe, b: Fe) -> Fe {
    let mut x = P as DoubleFe + a as DoubleFe - b as DoubleFe;
    if x >= P as DoubleFe {
        x -= P as DoubleFe;
    }
    x as Fe
}

pub fn generate_random_element() -> Fe {
    loop {
        let mut buf = [0u8; std::mem::size_of::<Fe>()];
        rand::thread_rng().fill_bytes(&mut buf);
        let num = Fe::from_le_bytes(buf);
        if num < Fe::MAX - (Fe::MAX % P) {
            return num % P;
        }
    }
}

pub fn modular_inverse(a: Fe) -> Fe {
    let mut result = Fe::from(1u8);
    let mut base = a;
    let mut exp = P - 2;

    while exp > 0 {
        if exp & 1 == 1 {
            result = mult(result, base);
        }
        base = mult(base, base);
        exp >>= 1;
    }
    result
}
