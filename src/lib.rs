//! Runtime spectral uplift for the Raygon spectral path tracer - the inference side of a learned
//! RGB/XYZ-to-reflectance model. (N-PEARL: Neural Predicted Emission And Reflectance from Lab.)
//!
//! # What this solves
//!
//! A spectral renderer integrates light transport over wavelength, so every material needs a reflectance
//! spectrum `$\rho(\lambda)$` - but art assets are authored as RGB/XYZ colours. Spectral uplift is the
//! inverse map: given a target colour, produce a plausible reflectance that reproduces it under a
//! reference illuminant. The map is underdetermined - infinitely many spectra integrate to the same
//! colour (metamers) - so the task is not to invert a function but to pick the metamer that behaves well
//! physically:
//!
//!   * colour-accurate under the authoring illuminant (small CIEDE2000 error);
//!   * smooth and bounded in `[0, 1]` (measured reflectances rarely spike);
//!   * stable under multiplication - an `n`-bounce path carries `$\rho^n$`, so a narrow notch goes black
//!     while a smooth curve darkens gracefully;
//!   * saturation-preserving under dimming, `$\text{uplift}(s\,c) \approx s\,\text{uplift}(c)$`.
//!
//! The model is trained offline against perceptual colour error with those properties as objectives;
//! this crate only runs the result. Inference happens per-texel on the CPU inside the BSDF hot loop,
//! where ray traversal already dominates, so the whole path is branch-light FMA over native SIMD width
//! and uses no transcendental functions - only `+ - * /` and `rsqrt`, with every nonlinearity algebraic.
//!
//! # How it works
//!
//! A small MLP (13K parameters) maps a colour - encoded as CIELAB plus the chromaticity of the
//! authoring illuminant's white, so chromatic adaptation is a learned relationship rather than a fixed
//! transform - to a short vector of reconstruction parameters. A fixed reconstructor turns those into the
//! base reflectance (see [`UpliftParams::reflectance`]): a low-degree Chebyshev series in wavelength plus
//! a Lorentzian resonance, summed in logit space, squashed into `[leak, 1-leak]` by an
//! algebraic sigmoid, then scaled. The polynomial carries the smooth trend; the resonance (the real
//! footprint of a single complex pole) adds the sharp, localized absorption band saturated colours need
//! without the ringing a high-degree polynomial would cause. Boundedness is structural - the sigmoid maps
//! any coefficients into range - and the constant scale gives exact homogeneity for the
//! saturation-preserving goal.
//!
//! Reflectance alone cannot reach the most saturated colours, which lie outside the reflective gamut. The
//! remaining parameters drive an energy-conserving rank-1 fluorescence term (absorb short, re-emit
//! Stokes-shifted long) that reaches them; see [`fluor`].
//!
//! The trained network is reparameterized for deploy ("train rich, ship plain"): train-only machinery is
//! folded into plain linear layers, and the weights are then quantized - against the appearance metric,
//! not raw reflectance, since perceptually identical metamers can differ in weight-space - and embedded as
//! a compact blob ([`DeployNet::load`]). The shipped network is ~13.5K parameters and reaches a round-trip
//! colour error around 0.02 mean dE2000, with in-gamut colours under 1 dE.
//!
//! # What is in this crate
//!
//! Two hot loops, both tight FMA over native-SIMD-width vectors:
//!   * [`DeployNet::uplift`] - the MLP forward (per material colour, once): output-stationary matvec,
//!     `acc = bias; for i in in { acc = W[:,i].mul_adde(splat(x[i]), acc) }` over `S::f32xN` chunks of
//!     the output dim, with the algebraic-SiLU gate between layers.
//!   * [`UpliftParams::reflectance`] - the per-wavelength reconstruction (per BSDF eval, hot): Chebyshev
//!     3-term recurrence + a Lorentzian resonance, FMA over the wavelength vector `V`, `alg_sigmoid4`
//!     bound via `inverse_sqrt`. Generic over any `V: NnVector` (the renderer's `$\lambda$` samples).
//!
//! Weight layout: each linear is stored column-major w.r.t. `(out, in)` - column `i` (an `out`-long
//! vector) is contiguous, so the matvec broadcasts `x[i]` and FMAs the contiguous column.

pub mod cube;
pub mod fluor;

use core::mem::MaybeUninit;

use thermite::prelude::*;

use thermite::math::RealMath;
use thermite_special::SpecialMath;

pub trait NnVector: SwizzleVector + RealMath + SpecialMath<Element = f32> {}
impl<V> NnVector for V where V: SwizzleVector + RealMath + SpecialMath<Element = f32> {}

pub trait NnBackend:
    SimdVectorsWithRegisters + SimdVectors<f32xN: NnVector, f32x4: NnVector, f32x8: NnVector, f32x16: NnVector>
{
}
impl<S> NnBackend for S where
    S: SimdVectorsWithRegisters + SimdVectors<f32xN: NnVector, f32x4: NnVector, f32x8: NnVector, f32x16: NnVector>
{
}

/// View an initialized prefix of a `MaybeUninit<f32>` buffer as `&[f32]` (stable stand-in for the unstable
/// `MaybeUninit::slice_assume_init_ref`). `MaybeUninit<f32>` is layout-identical to `f32`.
///
/// # Safety
/// Every element of `s` must be initialized.
#[inline(always)]
pub(crate) unsafe fn slice_assume_init_ref(s: &[MaybeUninit<f32>]) -> &[f32] {
    // SAFETY: caller guarantees all elements are initialized; the cast preserves length and layout.
    unsafe { &*(s as *const [MaybeUninit<f32>] as *const [f32]) }
}

/// Decode an IEEE-754 binary16 (half) to f32 (dependency-free). Values are always finite and in range, so
/// the only split is normal vs subnormal; the inf/NaN case is dropped (a `debug_assert` catches it).
///
/// Both paths are FP-free bit re-packs (the subnormal one needs a single multiply): the layouts differ only
/// by mantissa width (`$10 \to 23$` bits, a left shift of 13) and exponent bias (`$15 \to 127$`, i.e.
/// `$+112$`), with the sign bit moved from bit 15 to bit 31.
#[inline]
pub(crate) fn f16_to_f32(h: u16) -> f32 {
    let h = h as u32;
    let sign = (h & 0x8000) << 16; // bit 15 -> bit 31
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    debug_assert!(exp != 0x1f, "f16_to_f32: inf/NaN half {h:#06x}");

    if exp == 0 {
        // Zero or subnormal: value = mant * 2^-24, exactly representable as a (sub)normal f32.
        // mant == 0 yields a signed zero for free. `0x3380_0000` is the bit pattern of 2^-24.
        let mag = mant as f32 * f32::from_bits(0x3380_0000);
        return f32::from_bits(mag.to_bits() | sign);
    }

    // Normal (exp in 1..=30): rebias the exponent 15 -> 127 (+112) and widen the mantissa 10 -> 23 bits.
    f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13))
}

// ============================================================================
// Hard-coded architecture + reconstruction constants (specific to the shipped network)
// ============================================================================

/// Flat-MLP layer widths `[in, h1.., out]` of the shipped network: `6/32/32/32/48/64/80/16`.
const DIMS: [usize; 8] = [6, 32, 32, 32, 48, 64, 80, 16];
const N_LAYERS: usize = DIMS.len() - 1; // 7 linears
const N_IN: usize = DIMS[0]; // 6 (lab_illum_c2)
const N_OUT: usize = DIMS[N_LAYERS]; // 16
const MAX_W: usize = 80; // widest layer (scratch buffer size)

// Reconstruction constants (wavelength domain, leak floor, Chebyshev degree).
const LAMBDA_LO: f32 = 360.0;
const LAMBDA_HI: f32 = 800.0;
const LEAK: f32 = 1.0e-4;
const CHEB_K: usize = 6; // 7 coefficients (deg-6, single component)

// LINEAR Cheb map u = 2t-1, t = (lambda - lo)/(hi - lo), as a single FMA: u = lambda*SLOPE + INTERCEPT.
// The slope folds the 1/(hi-lo) divide into a compile-time constant (vs `rescale`, which divides per call).
const U_SLOPE: f32 = 2.0 / (LAMBDA_HI - LAMBDA_LO);
const U_INTERCEPT: f32 = -1.0 - LAMBDA_LO * U_SLOPE;

// Output-slot offsets in the 16-vector: [coeffs(7) | res x,y,a,b(4) | scale(1) | fluor c,le,s(3) | r(1)].
const O_COEFFS: usize = 0; // ..7
const O_RES: usize = 7; // ..11
// const O_SCALE: usize = 11; // 1
// const O_C: usize = 12;
// const O_LE: usize = 13;
// const O_S: usize = 14;
// const O_R: usize = 15;

// Output bounds applied to the network's raw output logits.
const RES_X_MAX: f32 = 1.2;
const RES_Y_MIN: f32 = 0.02;
const RES_Y_MAX: f32 = 1.0;
const RES_AMP_MAX: f32 = 12.0;
const LE_LO: f32 = 400.0;
const LE_HI: f32 = 700.0;
const S_LO: f32 = 5.0;
const S_HI: f32 = 150.0;
const R_MIN: f32 = 0.15;
const R_MAX: f32 = 1.75;

// CIELAB f() constants (white = E = (1,1,1)).
const LAB_EPS: f32 = (6.0 / 29.0) * (6.0 / 29.0) * (6.0 / 29.0); // (6/29)^3
const LAB_KAPPA: f32 = (29.0 / 3.0) * (29.0 / 3.0) * (29.0 / 3.0); // (29/3)^3

// ============================================================================
// Scalar algebraic activations / bounds (per-material; not the hot SIMD path)
// ============================================================================

// #[inline(always)]
// fn alg_sigmoid(x: f32) -> f32 {
//     0.5 + 0.5 * x / (1.0 + x * x).sqrt()
// }
// #[inline(always)]
// fn alg_sigmoid4(x: f32) -> f32 {
//     let x2 = x * x;
//     0.5 + 0.5 * x / (1.0 + x2 * x2).sqrt().sqrt() // (1 + x^4)^(1/4)
// }

#[inline(always)]
fn alg_tanh(x: f32) -> f32 {
    x / (1.0 + x * x).sqrt()
}

// #[inline(always)]
// fn alg_silu_s(x: f32) -> f32 {
//     x * alg_sigmoid(x)
// }

#[inline(always)]
fn lab_f(t: f32) -> f32 {
    if t > LAB_EPS {
        t.scalar_cbrt()
    } else {
        (LAB_KAPPA * t + 16.0) / 116.0
    }
}

/// CIELAB under the E white (1,1,1): xyz -> (L, a, b).
#[inline(always)]
fn xyz_to_lab_e(xyz: [f32; 3]) -> [f32; 3] {
    let (fx, fy, fz) = (lab_f(xyz[0]), lab_f(xyz[1]), lab_f(xyz[2]));
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

// ============================================================================
// SIMD kernels (FMA over native width)
// ============================================================================

/// Algebraic SiLU on a vector: `$a\,(0.5 + 0.5\,a\,(1 + a^2)^{-1/2})$`.
#[inline(always)]
fn alg_silu_v<V: NnVector>(a: V) -> V {
    let x = a.mul_adde(a, V::ONE);

    if const { V::HAS_APPROX_RSQRT } {
        let y = x.rsqrt();

        // Newton factor for 1/sqrt(x), 0.5 folded into the constants:
        //   nf = 0.75 - 0.25*x*y^2,  so (a*y)*nf ≈ 0.5*a/sqrt(x)
        let nf = y.square().mul_adde(x.scale(-0.25), V::splat(0.75));

        // silu = 0.5*a + 0.5*a^2/sqrt(x) = a*(a*y*nf + 0.5)
        a * (a * y).mul_adde(nf, V::HALF)
    } else {
        let g = a / x.sqrt(); // a * (1 + a^2)^(-1/2)
        a * g.mul_adde(V::HALF, V::HALF)
    }
}

/// Bound the trailing 8 reconstruction outputs in one SIMD pass over a native `V` (assumed 8-wide). The
/// raw logits `o[O_RES+1..]` are, in lane order, `[res.y, res.alpha, res.beta, scale, c, lambda_e, s, r]`.
///
/// `res.x` is not here: there are 9 bounded outputs total, one too many for an 8-lane vector, so the
/// cheapest one is peeled to a scalar by the caller. `res.x` is an `alg_tanh` lane (one sqrt, no bias, no
/// `[lo,hi]` remap) -> the cheapest activation, and it sits at the contiguous start (`O_RES`), so dropping
/// it lets the remaining 8 load as one contiguous vector from `O_RES+1`.
///
/// Every bound has the same shape `$\text{out} = A + B\,\text{core}_p(\text{raw})$`, where the shared
/// algebraic squashing core is `$\text{core}_p(x) = x\,(1 + x^p)^{-1/p}$`:
///   * `p = 2` -> `alg_sigmoid` (with the 0.5 bias/scale folded into A,B)
///   * `p = 4` -> `alg_sigmoid4`
///
/// The per-lane affine `(A, B)` folds in both the sigmoid bias/scale (`fb, fs`) and the `[lo, hi]` remap:
///   `A = lo + amp*fb`, `B = amp*fs`   (amp = hi-lo, or the symmetric amplitude for the alpha/beta lanes).
/// Only the core power differs across lanes, so a single mask `is_p4` selects the 4th-power core.
#[inline(always)]
fn bound_outputs<V: NnVector<Lanes = thermite::generic_array::typenum::U8>>(o: &[f32]) -> [f32; 8] {
    // raw logits, lane order [y, alpha, beta, scale, c, le, s, r]
    let x = unsafe { V::load_unaligned(o.as_ptr().add(O_RES + 1)) };

    const DY: f32 = RES_Y_MAX - RES_Y_MIN;
    const DLE: f32 = LE_HI - LE_LO;
    const DS: f32 = S_HI - S_LO;
    const DR: f32 = R_MAX - R_MIN;

    // A = lo + amp*fb   (fb: tanh 0, sigmoid/sigmoid4 0.5)
    let a = V::new(
        const {
            [
                RES_Y_MIN + 0.5 * DY, // y      sig,   lo RES_Y_MIN
                0.0,                  // alpha  tanh
                0.0,                  // beta   tanh
                0.5,                  // scale  sig4,  lo 0
                0.5,                  // c      sig4
                LE_LO + 0.5 * DLE,    // le     sig4,  lo LE_LO
                S_LO + 0.5 * DS,      // s      sig4,  lo S_LO
                R_MIN + 0.5 * DR,     // r      sig4,  lo R_MIN
            ]
        },
    );
    // B = amp*fs   (fs: tanh 1, sigmoid/sigmoid4 0.5)
    let b = V::new(
        const {
            [
                0.5 * DY,    // y
                RES_AMP_MAX, // alpha
                RES_AMP_MAX, // beta
                0.5,         // scale
                0.5,         // c
                0.5 * DLE,   // le
                0.5 * DS,    // s
                0.5 * DR,    // r
            ]
        },
    );

    // lanes 3..8 (scale,c,le,s,r) use the sharp 4th-power core; lanes 0..3 (y,alpha,beta) use 2nd-power.
    let is_p4 = V::new([0., 0., 0., 1., 1., 1., 1., 1.]).cmp_gt(V::ZERO);

    // core_p(x) = x * (1 + x^p)^(-1/p):
    //   base = 1 + x^2  (p=2)  |  1 + x^4  (p=4, via x2*x2)
    //   inv  = base^(-1/2) = inverse_sqrt(base)
    //   d    = inv  (p=2)  |  sqrt(inv) = base^(-1/4)  (p=4)
    let x2 = x * x;
    let base = is_p4.select(x2 * x2, x2) + V::ONE;
    let inv = base.inverse_sqrt();
    let d = is_p4.select(inv.sqrt(), inv);

    // out = A + B*core
    d.mul_adde(b * x, a).into_array().into_array()
}

/// Output-stationary `y = act?(W*x + b)`. `w` is column-major `(OUT, IN)`: column `i` at
/// `w[i*OUT .. (i+1)*OUT]`. Vectorized over the output dim in `V::LANES` chunks (FMA).
///
/// `IN`/`OUT`/`NCH`/`SPLIT` are const so everything unrolls at monomorphization (no dynamic dims, no
/// remainder loop, no dims-table / `Vec`-pointer reload per layer). `NCH = OUT / V::LANES` is the number
/// of output chunks; `SPLIT` is the input-reduction split (see below). With `V = f32x16` on AVX2 each
/// logical FMA double-pumps two ymm registers.
///
/// All output chunks are kept live and the input streamed once: each `x[i]` is broadcast once and FMA'd
/// into every chunk, so each input is broadcast once (not once per chunk) and the `NCH` chunk chains are
/// independent (ILP hides FMA latency -- the single-accumulator form was a serial chain bound by latency).
///
/// `SPLIT` widens that ILP when `NCH` alone is too few chains (e.g. the narrow last layer, `NCH=1` -> only
/// 2 ymm chains): it keeps `SPLIT` independent partial-accumulator sets, partial `s` summing inputs
/// `s, s+SPLIT, ...`, then combines them per chunk with a log-depth [`reduce_array`]. The bias seeds
/// partial 0 only. `SPLIT=1` is the zero-overhead no-op path for the wide layers. Requires `IN % SPLIT == 0`.
#[inline(always)]
fn matvec_act<V: NnVector, const ACT: bool, const IN: usize, const OUT: usize, const NCH: usize, const SPLIT: usize>(
    w: &[f32],
    b: &[f32],
    x: &[f32],
    y: &mut [MaybeUninit<f32>],
) {
    let lanes = V::LANES;
    debug_assert_eq!(NCH * lanes, OUT, "NCH * V::LANES must equal OUT");
    debug_assert_eq!(IN % SPLIT, 0, "SPLIT must divide IN");

    // acc[j][s] = chunk j, partial s. Repeat-init to ZERO (no array::from_fn), bias seeds partial 0.
    // Small + statically indexed after unrolling, so SROA keeps it in registers.
    let mut acc = [[V::ZERO; SPLIT]; NCH];
    let mut j = 0;
    while j < NCH {
        // SAFETY: j < NCH so j*lanes + lanes <= OUT <= len(b).
        acc[j][0] = unsafe { V::load_unaligned(b.as_ptr().add(j * lanes)) };
        j += 1;
    }

    let mut i = 0;
    while i < IN {
        let mut s = 0;
        while s < SPLIT {
            let xi = V::splat(x[i + s]);
            let base = (i + s) * OUT;
            let mut j = 0;
            while j < NCH {
                // SAFETY: (i+s) < IN and j < NCH, so base + j*lanes + lanes <= IN*OUT = len(w).
                let col = unsafe { V::load_unaligned(w.as_ptr().add(base + j * lanes)) };
                acc[j][s] = col.mul_adde(xi, acc[j][s]);
                j += 1;
            }
            s += 1;
        }
        i += SPLIT;
    }

    let mut j = 0;
    while j < NCH {
        // log-depth combine of the SPLIT partials -> acc[j][0]
        thermite::math::algorithms::reduce_in_place(
            &mut acc[j],
            #[inline(always)]
            |a, b| a + b,
        );

        let mut out = acc[j][0];

        if const { ACT } {
            out = alg_silu_v(acc[j][0])
        };

        // SAFETY: j < NCH so j*lanes + lanes <= OUT <= len(y). Storing through the MaybeUninit
        // backing as *mut f32 initializes lanes [j*lanes, j*lanes+lanes); across all j the loop writes
        // every slot of y[..OUT], so the caller may assume_init that prefix afterwards.
        unsafe { out.store_unaligned(y.as_mut_ptr().add(j * lanes).cast::<f32>()) };
        j += 1;
    }
}

// ============================================================================
// QNN quantized-weight decoder (dependency-free)
// ============================================================================

/// Max unary run before the Golomb-Rice escape (a raw 32-bit value follows). Must match the encoder.
const QNN1_ESCAPE_Q: u32 = 24;

/// `k` sentinel marking a mixed-precision channel: its fan-in weights are stored raw f32 in the tail
/// (not Rice-coded), so high-sensitivity "spike" channels stay exact and don't constrain the global step.
const QNN1_RAW_K: u32 = 0xFF;

/// Total bias count across all layers = sum(DIMS[1..]); the raw tail is biases, then whitener, then any
/// raw-f32 channel weights.
const N_BIAS: usize = {
    let mut s = 0;
    let mut l = 1;
    while l < DIMS.len() {
        s += DIMS[l];
        l += 1;
    }
    s
};

#[inline]
fn unzigzag(z: u32) -> i32 {
    ((z >> 1) as i32) ^ -((z & 1) as i32)
}

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize, // bit position
}
impl<'a> BitReader<'a> {
    #[inline]
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
    #[inline]
    fn get_bit(&mut self) -> u32 {
        let bit = (self.data[self.pos >> 3] >> (7 - (self.pos & 7))) & 1;
        self.pos += 1;
        bit as u32
    }
    #[inline]
    fn get_bits(&mut self, n: u32) -> u32 {
        let mut v = 0u32;
        for _ in 0..n {
            v = (v << 1) | self.get_bit();
        }
        v
    }
}

#[inline(always)]
fn rice_decode(br: &mut BitReader, k: u32) -> u32 {
    let mut q = 0u32;
    while br.get_bit() == 0 {
        q += 1;
        if q == QNN1_ESCAPE_Q {
            return br.get_bits(32); // escaped raw value (no terminating 1)
        }
    }
    let r = if k > 0 { br.get_bits(k) } else { 0 };
    (q << k) | r
}

// ============================================================================
// Loaded network + bounded per-material parameters
// ============================================================================

/// The folded deploy network: column-major weights + biases per layer (owned; ~48 KB), plus the
/// input-feature standardization stats. Loaded once per scene.
#[derive(Clone, Debug)]
pub struct DeployNet {
    w: [Vec<f32>; N_LAYERS], // [layer] column-major (out, in)
    b: [Vec<f32>; N_LAYERS],
    whit_mean: [f32; N_IN],
    whit_std: [f32; N_IN],
}

/// Bounded reconstruction parameters for one material colour. The base reflectance is evaluated by
/// [`Self::reflectance`]; the fluorescence fields `(c, lambda_e, s, r)` are consumed by the renderer's
/// energy-conserving fluor forward.
#[derive(Clone, Copy, Debug)]
pub struct UpliftParams {
    coeffs: [f32; CHEB_K + 1],
    // Lorentzian resonance, PRECOMPUTED (lambda-independent) for the hot path. The logit term is
    // `res_a/den + res_b*d/den`, `den = d^2 + res_yy`, `d = u - res_x` (see `reflectance`).
    res_x: f32,  // center, in u-space
    res_yy: f32, // y^2 (denominator floor + absorption weight)
    res_a: f32,  // alpha * y^2 (absorption numerator)
    res_b: f32,  // beta * y    (dispersion numerator coefficient)
    // Output sigmoid + leak + global scale collapsed: `rho = tail_c0 + tail_c1 * (p*q)`.
    tail_c0: f32, // 0.5 * scale
    tail_c1: f32, // 0.5 * scale * (1 - 2*leak)
    /// Fluorophore concentration / mix `$c \in [0, 1]$`.
    pub c: f32,
    /// Emission peak `lambda_e` (nm).
    pub lambda_e: f32,
    /// Stokes shift `s` (nm).
    pub s: f32,
    /// Emission-width ratio `r`.
    pub r: f32,
}

impl DeployNet {
    /// Decode the embedded compressed weight blob (`model/deploy.qnn`) into the column-major layer
    /// buffers. The weights are post-training quantized and Golomb-Rice coded; the quantization is
    /// optimized against the appearance metric (CIELAB under D65 / dE2000) rather than raw reflectance,
    /// since perceptually identical metamers can differ in weight-space. Dependency-free: one linear pass
    /// of integer/bit ops plus one multiply per weight.
    ///
    /// Container: `magic u32 | n_ch u32 | [scale f16, k u8] x n_ch | n_f16 u32 | f16 x n_f16 |
    /// n_f32 u32 | f32 x n_f32 | Rice payload`. A channel is an output neuron (its weights = its fan-in);
    /// the fan-in is recovered from `DIMS`, not stored. Channel order is layer 0 neurons, layer 1 neurons, ...
    /// so each row scatters into the column-major buffer at `i*out_d + c`. The `f16` section is biases
    /// (layer order) then the fan-in weights of every mixed-precision (`k == QNN1_RAW_K`) channel; the `f32`
    /// section is the input whitener mean/std (kept exact). Per-channel scales and biases are fp16, and the
    /// integer weights are rounded against that exact fp16 step so the rounding is compensated.
    pub fn load() -> Self {
        let blob = include_bytes!("../model/deploy.qnn");

        let rd_u32 = |o: usize| u32::from_le_bytes([blob[o], blob[o + 1], blob[o + 2], blob[o + 3]]);
        let rd_f32 = |o: usize| f32::from_le_bytes([blob[o], blob[o + 1], blob[o + 2], blob[o + 3]]);
        let rd_f16 = |o: usize| f16_to_f32(u16::from_le_bytes([blob[o], blob[o + 1]]));
        assert_eq!(rd_u32(0), 0x514E_4E32, "bad QNN2 magic");
        let n_ch = rd_u32(4) as usize;

        // per-channel (scale f16, k u8) = 3 bytes. k == QNN1_RAW_K -> raw channel (weights in the f16 section).
        let mut chans: Vec<(f32, u32)> = Vec::with_capacity(n_ch);
        let mut off = 8;
        for _ in 0..n_ch {
            let scale = rd_f16(off);
            let k = blob[off + 2] as u32;
            off += 3;
            chans.push((scale, k));
        }

        // f16 section: biases (N_BIAS, layer order) then raw-channel weights.
        let n_f16 = rd_u32(off) as usize;
        off += 4;
        let mut f16v = Vec::with_capacity(n_f16);
        for i in 0..n_f16 {
            f16v.push(rd_f16(off + 2 * i));
        }
        off += 2 * n_f16;

        // f32 section: whitener mean/std.
        let n_f32 = rd_u32(off) as usize;
        off += 4;
        let mut f32v = Vec::with_capacity(n_f32);
        for i in 0..n_f32 {
            f32v.push(rd_f32(off + 4 * i));
        }
        off += 4 * n_f32;

        // Running pointer into the f16 section for raw-channel weights (after the biases).
        let mut raw_w = N_BIAS;

        // Rice payload (+ f16 raw weights) -> weights, scattered into column-major (in, out) layer buffers.
        let mut br = BitReader::new(&blob[off..]);
        let mut w: [Vec<f32>; N_LAYERS] = Default::default();
        let mut ch = 0;
        for l in 0..N_LAYERS {
            let (in_d, out_d) = (DIMS[l], DIMS[l + 1]);
            w[l] = vec![0.0; in_d * out_d];
            for c in 0..out_d {
                let (scale, k) = chans[ch];
                ch += 1;
                if k == QNN1_RAW_K {
                    for i in 0..in_d {
                        w[l][i * out_d + c] = f16v[raw_w];
                        raw_w += 1;
                    }
                } else {
                    for i in 0..in_d {
                        let q = unzigzag(rice_decode(&mut br, k));
                        w[l][i * out_d + c] = q as f32 * scale;
                    }
                }
            }
        }
        debug_assert_eq!(ch, n_ch, "channel count");
        debug_assert_eq!(raw_w, n_f16, "f16 section fully consumed");

        // biases from the f16 section (per layer); whitener from the f32 section (input standardization).
        let mut b: [Vec<f32>; N_LAYERS] = Default::default();
        let mut bi = 0;
        for l in 0..N_LAYERS {
            b[l] = f16v[bi..bi + DIMS[l + 1]].to_vec();
            bi += DIMS[l + 1];
        }
        let mut whit_mean = [0.0f32; N_IN];
        let mut whit_std = [1.0f32; N_IN];
        for i in 0..N_IN {
            whit_std[i] = 1.0 / f32v[N_IN + i];
            whit_mean[i] = f32v[i] * whit_std[i];
        }

        Self {
            w,
            b,
            whit_mean,
            whit_std,
        }
    }

    /// Write the 6 standardized `lab_illum_c2` input features for a material colour (XYZ under its
    /// authoring illuminant) into `out[..N_IN]`. `white_ab` is the E-referenced CIELAB `(a, b)` of that
    /// illuminant's white -- precomputed once per illuminant by [`Self::white_lab_ab`] (the white's `L` is
    /// always ~100 and unused, so its 3 cbrts never hit the per-material path).
    ///
    /// The colour's XYZ->Lab `f()` (`t>eps ? cbrt(t) : (kappa*t+16)/116`) is done in one 4-lane SIMD pass
    /// (`V = f32x4`, lane 3 unused): a single vector `cbrt` + blend replaces the 3 scalar f64-division cbrts.
    #[inline(always)]
    fn input_features<V: NnVector<Lanes = thermite::generic_array::typenum::U4>>(
        &self,
        color_xyz: [f32; 3],
        white_ab: [f32; 2],
        out: &mut [MaybeUninit<f32>],
    ) {
        let v = V::new([color_xyz[0], color_xyz[1], color_xyz[2], 0.0]); // lane 3 unused (w = 0)
        let big = v.cmp_gt(V::splat(LAB_EPS));
        let lin = v.mul_adde(V::splat(LAB_KAPPA / 116.0), V::splat(16.0 / 116.0)); // (kappa*t + 16)/116
        let f = big.select(v.cbrt(), lin); // f = [fx, fy, fz, *]

        // CIELAB combine fully in-register (avoids extracting fx,fy,fz to scalars): each output lane is a
        // 2-term linear combo of f, so  lab = COEF*swizzle(f) - (COEFS*f + CST)  with one swizzle + 2 FMAs:
        //   swizzle(f)=[fy, fx, fy];  COEF=[116,500,200];  COEFS=[0,500,200];  CST=[16,0,0]
        //   -> lane0 = 116*fy - 16 = L;  lane1 = 500*(fx-fy) = a;  lane2 = 200*(fy-fz) = b.
        // COEFS[0]=0 zeroes f's contribution to L, so the unused lane 3 never matters.
        let p = thermite::swizzle!(f, [1, 0, 1, 1]); // [fy, fx, fy, _]
        let sub = V::new(const { [0.0, 500.0, 200.0, 0.0] }).mul_adde(f, V::new(const { [16.0, 0.0, 0.0, 0.0] }));
        let lab = V::new(const { [116.0, 500.0, 200.0, 0.0] }).mul_sube(p, sub);
        let [lab_l, lab_a, lab_b, _] = lab.into_array().into_array();

        let dx = lab_a - white_ab[0];
        let dy = lab_b - white_ab[1];
        let c2 = dx.mul_adde(dx, dy * dy);

        // Standardize (raw - mean)/std = raw*inv_std - mean*inv_std (whit_std holds inv_std, whit_mean the
        // premultiplied mean). Written directly -- no `raw` array, no loop.
        // Writes out[0..N_IN] in full (the caller assume_inits exactly that prefix).
        let (m, s) = (&self.whit_mean, &self.whit_std);
        out[0].write(lab_l.mul_sube(s[0], m[0]));
        out[1].write(lab_a.mul_sube(s[1], m[1]));
        out[2].write(lab_b.mul_sube(s[2], m[2]));
        out[3].write(white_ab[0].mul_sube(s[3], m[3]));
        out[4].write(white_ab[1].mul_sube(s[4], m[4]));
        out[5].write(c2.mul_sube(s[5], m[5]));
    }

    /// E-referenced CIELAB `(a, b)` of an illuminant white XYZ, for [`Self::uplift`]. Compute once per
    /// illuminant (not per material).
    #[inline]
    pub fn white_lab_ab(white_xyz: [f32; 3]) -> [f32; 2] {
        let lab = xyz_to_lab_e(white_xyz);
        [lab[1], lab[2]]
    }
}

type F32x4<S> = <S as SimdVectors>::f32x4;
type F32x8<S> = <S as SimdVectors>::f32x8;
type F32x16<S> = <S as SimdVectors>::f32x16;

impl DeployNet {
    /// MLP forward pass -> the 16 raw network outputs (pre-bounding). Shared by the dispatched entry
    /// points [`Self::uplift`] / [`Self::uplift_raw`]; `#[inline(always)]` so it folds into their
    /// `#[target_feature]` bodies and the matvec FMAs use the dispatched ISA.
    #[inline(always)]
    fn forward<S: NnBackend>(&self, color_xyz: [f32; 3], white_ab: [f32; 2]) -> [f32; N_OUT] {
        // Double-width scratch split into two halves; each layer reads `src`, writes `dst`, then swaps the
        // two &mut references (pointer swap, no copy). After N_LAYERS (=7, odd) swaps the output is in `src`.
        //
        // Left uninitialized (no per-call zeroing memset): every slot is written before it is read.
        // `input_features` writes `src[..N_IN]`; then layer L reads only `src[..DIMS[L]]` (fully written by
        // layer L-1, or by input_features for L=0) and writes all of `dst[..DIMS[L+1]]`, seeding its
        // accumulators from `b` rather than from `dst`. So we hand each writer a `&mut [MaybeUninit<f32>]`
        // target and only ever form a `&[f32]` read-view over the prefix that was just initialized.
        let mut scratch: [MaybeUninit<f32>; 2 * MAX_W] = [const { MaybeUninit::uninit() }; 2 * MAX_W];
        let (mut src, mut dst) = scratch.split_at_mut(MAX_W);

        self.input_features::<F32x4<S>>(color_xyz, white_ab, &mut src[..N_IN]);
        // SAFETY: input_features wrote all of src[..N_IN]; that prefix is now initialized.
        let mut src_len = N_IN;

        // Layers unrolled with const dims (single source of truth = DIMS) so each matvec monomorphizes to
        // a fixed shape, over f32x16 (double-pumped AVX2). NCH = OUT/16 output chunks held live for ILP.
        // After each, swap the src/dst references.
        macro_rules! layer {
            ($l:expr, $act:expr) => {{
                const NCH: usize = DIMS[$l + 1] / 16;
                // Target ~4 f32x16 accumulators (~8 ymm) for full ILP: split the input reduction only when
                // NCH alone gives too few chains. NCH>=3 already saturates, so SPLIT=1 (no-op) there.
                const SPLIT: usize = if NCH >= 3 { 1 } else { 4 / NCH };
                debug_assert_eq!(src_len, DIMS[$l], "src prefix length must equal this layer's IN");
                // SAFETY: src[..DIMS[$l]] was fully written by the previous writer (input_features for
                // l=0, the prior layer otherwise), tracked by `src_len`.
                let xin: &[f32] = unsafe { slice_assume_init_ref(&src[..DIMS[$l]]) };
                matvec_act::<F32x16<S>, $act, { DIMS[$l] }, { DIMS[$l + 1] }, NCH, SPLIT>(
                    &self.w[$l],
                    &self.b[$l],
                    xin,
                    &mut dst[..DIMS[$l + 1]],
                );
                // dst[..DIMS[$l+1]] is now fully initialized; after the swap it becomes the new src prefix.
                core::mem::swap(&mut src, &mut dst);
                src_len = DIMS[$l + 1];
            }};
        }
        layer!(0, true);
        layer!(1, true);
        layer!(2, true);
        layer!(3, true);
        layer!(4, true);
        layer!(5, true);
        layer!(6, false); // last layer: bare linear readout

        debug_assert_eq!(src_len, N_OUT, "final src prefix length must equal N_OUT");
        let mut out = [0.0f32; N_OUT];
        // SAFETY: the last layer wrote all of src[..N_OUT] (DIMS[N_LAYERS] = N_OUT).
        out.copy_from_slice(unsafe { slice_assume_init_ref(&src[..N_OUT]) });
        out
    }
}

#[thermite::dispatch(S)]
impl DeployNet {
    /// Run the MLP and bound the outputs into reconstruction parameters. `S` selects the native vector
    /// width for the matvec FMA loops. `white_ab` is the illuminant white's E-referenced CIELAB `(a, b)`
    /// from [`Self::white_lab_ab`] (precompute once per illuminant).
    pub fn uplift<S: NnBackend>(&self, color_xyz: [f32; 3], white_ab: [f32; 2]) -> UpliftParams {
        UpliftParams::from_outputs::<S>(&self.forward::<S>(color_xyz, white_ab))
    }

    /// The 16 raw network outputs (pre-bounding), for baking a [`crate::cube::LookupCube`] -- the same
    /// forward as [`Self::uplift`] without the final bound/precompute step.
    pub fn uplift_raw<S: NnBackend>(&self, color_xyz: [f32; 3], white_ab: [f32; 2]) -> [f32; N_OUT] {
        self.forward::<S>(color_xyz, white_ab)
    }
}

impl UpliftParams {
    /// Bound + precompute the reconstruction parameters from the network's 16 raw outputs (slot order
    /// `[coeffs(7) | res x,y,alpha,beta(4) | scale | c, lambda_e, s | r]`). Shared by the live MLP path
    /// ([`DeployNet::uplift`]) and the [`crate::cube`] lookup. `res.x` is peeled to a scalar `alg_tanh`
    /// (cheapest lane: no bias/remap); the other 8 bounds run in one masked SIMD pass.
    #[inline]
    pub(crate) fn from_outputs<S: NnBackend>(o: &[f32]) -> Self {
        let res_x = RES_X_MAX * alg_tanh(o[O_RES]);
        let bnd = bound_outputs::<F32x8<S>>(o);
        let (y, alpha, beta, scale) = (bnd[0], bnd[1], bnd[2], bnd[3]);

        let mut coeffs = [0.0f32; CHEB_K + 1];
        coeffs.copy_from_slice(&o[O_COEFFS..O_COEFFS + CHEB_K + 1]);

        // lambda-independent reflectance constants, hoisted out of the per-eval hot path.
        let yy = y * y;
        Self {
            coeffs,
            res_x,
            res_yy: yy,
            res_a: alpha * yy,
            res_b: beta * y,
            tail_c0: scale * 0.5,
            tail_c1: scale * const { 0.5 * (1.0 - 2.0 * LEAK) },
            c: bnd[4],
            lambda_e: bnd[5],
            s: bnd[6],
            r: bnd[7],
        }
    }

    /// Base reflectance `$\rho(\lambda)$` in `[leak, 1-leak] * scale`, vectorized over the wavelength
    /// vector `V` (the hot path): linear-mapped Chebyshev (deg-6) + one Lorentzian resonance, through
    /// the sharp algebraic sigmoid. `lambda` in nm. Does not include fluorescence (that is the renderer's
    /// energy-conserving forward, which consumes `c, lambda_e, s, r`).
    #[inline(always)]
    pub fn reflectance<V: NnVector>(&self, lambda: V) -> V {
        // LINEAR Cheb map u = 2t-1 as one FMA (U_SLOPE folds in the 1/(hi-lo) divide). The Uber path uses
        // the LINEAR basis -- train_uber/uber_viz build the ChebReconstructor with no warp arg, so the
        // `cheb_smoothstep_warp` config flag is a no-op there (only legacy viz_common honored it).
        let u = lambda.mul_adde(V::splat(U_SLOPE), V::splat(U_INTERCEPT));

        // // Smoothstep warp: t in [0,1], s = t^2(3 - 2t), u = 2s - 1 in [-1,1].
        // let s = lambda.smoothstep::<2>(Some((V::splat(LAMBDA_LO), V::splat(LAMBDA_HI))));

        let mut p = u.chebyshev::<1, _>(&self.coeffs);

        // Lorentzian resonance with the lambda-independent terms hoisted into UpliftParams. Both terms
        // share 1/den, so the numerator is one FMA: logit += (res_a + res_b*d)/den, den = d^2 + res_yy.
        let d = u - V::splat(self.res_x);
        let inv = d.mul_adde(d, V::splat(self.res_yy)).reciprocal(); // 1/(d^2 + y^2); den >= y_min^2 > 0
        let num = V::splat(self.res_b).mul_adde(d, V::splat(self.res_a)); // res_b*d + res_a
        p = num.mul_adde(inv, p);

        // alg_sigmoid4 + leak + scale collapsed: rho = tail_c0 + tail_c1 * (p * q), q = (1 + p^4)^(-1/4).
        let p2 = p * p;
        let q = p2.mul_adde(p2, V::ONE).sqrt().inverse_sqrt();
        V::splat(self.tail_c1).mul_adde(p * q, V::splat(self.tail_c0))
    }
}
