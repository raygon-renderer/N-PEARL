//! Energy-conserving rank-1 fluorescence for the spectral path tracer (Jung 2019 model, Mojzík 2018
//! bispectral handling). The fluorophore parameters `$(c, \lambda_e, s, r)$` come from [`UpliftParams`]
//! (the neural uplift); this module turns them into the analytic absorption/emission profiles and the
//! reradiation operators a BSDF will call. Constants are hard-coded to the shipped network.
//!
//! # The model
//!
//! Reflectance alone cannot reach the most saturated colours - they lie outside the reflective gamut.
//! Fluorescence reaches them by adding energy, inelastically: light absorbed at a (usually shorter)
//! excitation wavelength `$\lambda_i$` is re-emitted at a (longer) emission wavelength `$\lambda_o$`, the
//! Stokes shift. The surface response is a reradiation matrix `$M(\lambda_o, \lambda_i)$`, not a diagonal
//! reflectance. By Kasha's rule the emission shape is independent of which wavelength excited it, so the
//! fluorescent part is rank-1 (separable):
//!
//! ```math
//! \begin{aligned}
//!   L_\text{out}(\lambda_o) ={}& (1 - c\,a(\lambda_o))\,\rho(\lambda_o)\,L_\text{in}(\lambda_o)
//!     && \text{elastic diagonal (attenuated reflectance)} \\
//!   {}+{}& Q\,c\,e(\lambda_o)\int a(\lambda_i)\,L_\text{in}(\lambda_i)\,d\lambda_i
//!     && \text{fluorescent rank-1 (absorbed power} \times \text{emission)}
//! \end{aligned}
//! ```
//!
//! * `$a(\lambda)$` - peak-normalized symmetric Lorentzian centered at `$\mu_a = \lambda_e - s$` (peak 1, in
//!   `[0,1]`).
//! * `$e(\lambda)$` - red-skewed split-Lorentzian centered at `$\lambda_e$`, hard Stokes truncation
//!   `$\lambda > \mu_a$`, normalized so `$\int e\,d\lambda = 1$`. Blue-flank HWHM `$\gamma r$`, red-flank
//!   `$\gamma\, r\, \text{skew}$`.
//! * `$\gamma$` (absorption HWHM) `$= \max(0.5\beta, 2)$`, Jung bandwidth `$\beta = \lambda_e\, s / (2\lambda_e - s)$`.
//! * `Q = 0.96` quantum yield. The `$(1 - c\,a)$` reuses the same `a` as the integral, so the per-excitation
//!   column sum `$(1 - c\,a)\,\rho + Q\,c\,a \le 1$` holds structurally (`$c, a \in [0,1]$`, `Q < 1`):
//!   energy-conserving.
//!
//! Normalizations are analytic (closed-form Lorentzian integral `$\gamma \cdot [\arctan\dots]$`), not grid sums, so every
//! profile is evaluable pointwise at the sparse SMIS wavelengths.
//!
//! # Two wavelength domains (note: UV)
//!
//! The shipped network was trained with a UV pump enabled, so excitation reaches into the UV (down to
//! 340 nm). Kasha makes the pump absorption-only: UV light is absorbed and down-converted to visible
//! emission (the observer is blind below 380). So:
//!
//! * Excitation / absorption uses [`EXCITATION_LO`]`..`[`EXCITATION_HI`]` = [340, 780]`, UV-inclusive. The
//!   excitation-shift sampling and any absorbed-power integral must reach into the UV, or UV-pumped
//!   materials (FWA, blacklight glow) are under-excited - exactly the materials the model was tuned for. Scene
//!   illuminants therefore need UV content for blacklight/FWA to render.
//! * Emission uses [`EMISSION_LO`]`..`[`EMISSION_HI`]` = [380, 780]`, visible only (emission, its normalization,
//!   and the `$(1 - c\,a)\,\rho$` attenuation are untouched by UV).
//!
//! # Rendering it: the bispectral wavelength shift (Mojzík 2018)
//!
//! A path tracer never evaluates `$\int a\,L_\text{in}$` for indirect light (it carries only the 4 SMIS wavelengths and
//! has no incident spectrum to integrate). Instead fluorescence is a stochastic wavelength shift at the
//! surface event, importance-sampled by the reradiation matrix:
//!
//! * Camera path (carrying `$\lambda_o$`, tracing backward): at a fluorescent hit, sample the incident
//!   `$\lambda_i$` from row `$\lambda_o$`. With probability `P_diag = elastic / (elastic + fluor)` take the
//!   diagonal (`$\lambda_i = \lambda_o$`, ordinary `$(1 - c\,a)\,\rho$` reflectance); otherwise draw
//!   `$\lambda_i \sim a(\lambda_i)$` over the excitation band ([`FluorProfile::sample_excitation`]) and continue the path at
//!   `$\lambda_i$`. The absorbed-power integral emerges from this MC estimate; you never integrate it. Row
//!   weights: `$\text{elastic} = (1 - c\,a(\lambda_o))\,\rho(\lambda_o)$`, `$\text{fluor} = Q\,c\,e(\lambda_o)\,A$`,
//!   `$A = \int_\text{excitation} a(\lambda_i)\,d\lambda_i$` (= `1/`[`FluorProfile::excite_pdf_norm`]).
//! * Next-event estimation (direct light): sample the shift first ([`FluorProfile::sample_emission`] gives the
//!   `$\lambda_o$` for a known excitation, or invert for `$\lambda_i$`), then sample the emitter at that wavelength -
//!   otherwise non-shifting/longer bands are unreachable and NEE returns zero on fluorescent surfaces.
//!
//! ## SMIS independent wavelength shift (the `$|\Lambda|^{n-1}$` correction) - integrator contract
//!
//! The `n = 4` SMIS lanes cannot shift together. Fluorescence localizes energy into one absorption and
//! one emission band; shifting all lanes by the same `$\Delta\lambda$` lands the non-hero lanes in
//! zero-transport regions and collapses the estimator to monochrome. So every lane shifts independently -
//! each lane `k` draws its own `$\lambda_{i,k}$` from its own `$\lambda_{o,k}$` via [`FluorProfile::sample_excitation`].
//!
//! Independent shifting enlarges the integration domain by a spurious factor `$|\Lambda|^{n-1}$` (`n-1` extra
//! free wavelengths). The unbiased correction is a per-lane throughput factor equal to the product of the
//! other lanes' shift PDFs:
//!
//! ```math
//! \bar{t}_k = \prod_{t \ne k} p(\lambda_{i,t} \mid \lambda_{o,t})
//! ```
//!
//! (lane `k` multiplied by the other lanes' shift pdfs.) When the MC sample is divided by the full joint pdf
//! `$p(\bar\lambda_i) = \prod_t p(\lambda_{i,t} \mid \lambda_{o,t})$`, the `$\bar{t}_k$` numerator cancels all
//! but lane `k`'s own pdf, leaving the correct per-lane `$p(\lambda_{i,k})$` and removing the
//! `$|\Lambda|^{n-1}$` domain factor. With [`FluorProfile::sample_excitation`] returning each lane's `$(\lambda_i, \text{pdf})$`:
//!
//! ```text
//! let prod = pdf.product_across_lanes();              // prod_t p_t  (one horizontal product over the vector)
//! throughput_k *= reradiation_value_k * (prod / pdf_k);   // = reradiation_k * prod_{t!=k} p_t  ... then / p_k
//! ```
//!
//! i.e. multiply each lane by `$\prod_{t \ne k} p_t$` (= `prod / pdf_k`) and keep dividing the lane's own
//! contribution by `p_k` as usual. After the first shift, lanes may leave the visible range; that is fine,
//! only the wavelength at the sensor matters for image reconstruction.
//!
//! ## MIS across shift strategies - integrator contract
//!
//! Each fluorescent vertex has multiple ways to have been sampled (diagonal vs off-diagonal; camera-side
//! BSDF shift vs light-side NEE shift; and, as in elastic SMIS, which lane was hero). Combine them with
//! the balance heuristic over the per-path probabilities, exactly as for elastic SMIS, treating the
//! shift pdf as another factor of the path pdf:
//!
//! ```math
//! w_\text{strategy} = \frac{p_\text{strategy}(\text{path})}{\sum_j p_j(\text{path})}
//! ```
//!
//! where each `p_j` includes the lane shift pdfs `$p(\lambda_{i,t} \mid \lambda_{o,t})$`. The pdfs needed for
//! these denominators are [`FluorProfile::sample_excitation`]'s returned pdf, [`FluorProfile::excite_pdf`] (evaluate the excitation pdf
//! for a given `$\lambda_i$`, for the other strategies' denominators), and [`FluorProfile::emission`] (which is itself the
//! normalized outgoing pdf, so [`FluorProfile::emission`] at `$\lambda_o$` doubles as `$p(\lambda_o \mid \lambda_i)$` for the
//! light-side strategy). Light/camera connection requires re-evaluating the light path's throughput at the
//! camera wavelengths (gated behind the shadow ray); the reradiation here is cheap to re-evaluate because the
//! profile is a closed form of the cached `$(c, \lambda_e, s, r)$`.

use thermite::math::TranscendentalMath;
use thermite::prelude::*;

use crate::{NnVector, UpliftParams};

// ============================================================================
// Hard-coded fluorescence constants (specific to the shipped network)
// ============================================================================

/// Fluorescence quantum yield (fraction of absorbed power re-emitted).
const Q: f32 = 0.96;
/// Absorption Lorentzian HWHM `$= \text{clamp}(\texttt{W\_SCALE} \cdot \beta, \texttt{W\_FLOOR}, \infty)$`, `$\beta = \lambda_e\, s / (2\lambda_e - s)$`.
const W_SCALE: f32 = 0.5;
const W_FLOOR: f32 = 2.0;
/// Emission red-flank widening (skew), ramped from `SKEW_BLUE` (short `$\lambda_e$`) to `SKEW_RED` (long
/// `$\lambda_e$`) linearly over `$\lambda_e \in [\texttt{SKEW\_LE0}, \texttt{SKEW\_LE1}]$`.
const SKEW_BLUE: f32 = 1.2;
const SKEW_RED: f32 = 1.4;
const SKEW_LE0: f32 = 450.0;
const SKEW_LE1: f32 = 500.0;

/// Emission (visible) band: normalization + outgoing-shift sampling range.
pub const EMISSION_LO: f32 = 380.0;
pub const EMISSION_HI: f32 = 780.0;
/// Excitation (absorption) band: UV-inclusive (down to 340 nm); see the module docs.
pub const EXCITATION_LO: f32 = 340.0;
pub const EXCITATION_HI: f32 = 780.0;

// ============================================================================
// Per-material fluorescence profile (precomputed once, like the reflectance hoist)
// ============================================================================

/// Analytic fluorophore profile for one material, derived once from [`UpliftParams`] via
/// [`FluorProfile::from_uplift`]. All `$\lambda$`-independent quantities (centers, HWHMs, the analytic emission
/// normalization, the absorption-sampling normalization) are precomputed so the per-wavelength evals and
/// the per-lane shift sampling are cheap. `c == 0` is the non-fluorescent fast path.
#[derive(Clone, Copy, Debug)]
pub struct FluorProfile {
    /// Fluorophore concentration / mix `$c \in [0, 1]$` (0 = non-fluorescent).
    pub c: f32,
    /// Absorption center `mu_a = lambda_e - s` (nm) - sampler center + Stokes mask.
    mu_a: f32,
    /// Absorption HWHM `gamma` (nm) - sampler scale.
    gamma_a: f32,
    /// `1/gamma` - so `absorption`'s `(lambda - mu_a)/gamma` is a multiply (no divide; subtract stays first
    /// to avoid catastrophic cancellation).
    inv_gamma_a: f32,
    /// `Q*c` - precomputed reradiation scale.
    qc: f32,
    /// Emission center `lambda_e` (nm).
    lambda_e: f32,
    /// Emission blue-/red-flank HWHM `gamma*r`, `gamma*r*skew` (nm) - sampler scales.
    gamma_b: f32,
    gamma_r: f32,
    /// Their reciprocals - so `emission`'s `(lambda - lambda_e)/gamma` becomes a multiply (no divide).
    inv_gamma_b: f32,
    inv_gamma_r: f32,
    /// `$1 / \int_\text{emission} e$` - analytic emission normalization (split-Lorentzian, Stokes-truncated).
    e_norm_rcp: f32,
    /// Emission inverse-CDF blue-flank lower angle `atan((blue_lo - lambda_e)/gamma_b)` (negative) and
    /// red-flank upper angle `atan((EMISSION_HI - lambda_e)/gamma_r)`, plus the blue-flank mass fraction - all
    /// lane-independent, shared by the emission normalization and `sample_emission` (which then needs no `atan`).
    emis_a_blo: f32,
    emis_a_rhi: f32,
    emis_blue_frac: f32,
    /// Excitation inverse-CDF lower angle `atan((EXCITATION_LO - mu_a)/gamma)` (lambda/lane-independent; precomputed
    /// so `sample_excitation` needs no `atan`).
    excite_a_lo: f32,
    /// Excitation inverse-CDF angular span `atan((EXCITATION_HI - mu_a)/gamma) - excite_a_lo`. The excitation integral is `gamma*excite_da`.
    excite_da: f32,
    /// The excitation integral `gamma*excite_da` (= `A`) and its reciprocal - precomputed so `sample_excitation`,
    /// `excite_pdf`, and the integrator row weight need no per-call multiply/reciprocal.
    excite_int: f32,
    excite_rcp: f32,
}

#[thermite::dispatch(S)]
impl FluorProfile {
    /// Derive the analytic profile from the NN uplift parameters (once per material). `S` only selects the
    /// 4-lane backend used to batch the four precompute `atan`s.
    pub fn from_uplift<S: SimdVectors<f32x4: TranscendentalMath>>(p: &UpliftParams) -> Self {
        Self::new::<S>(p.c, p.lambda_e, p.s, p.r)
    }

    /// Derive the analytic profile from the raw bounded fluorophore parameters `$(c, \lambda_e, s, r)$`. `S` selects
    /// the 4-lane backend used to evaluate the four lane-independent inverse-CDF angles in a single `f32x4`
    /// divide + `atan`.
    pub fn new<S: SimdVectors<f32x4: TranscendentalMath>>(c: f32, lambda_e: f32, s: f32, r: f32) -> Self {
        let mu_a = lambda_e - s;

        // Jung-bandwidth-derived absorption HWHM. 2*lambda_e - s > 0 over the head ranges (lambda_e >= 400, s <= 150).
        let beta = lambda_e * s / (2.0 * lambda_e - s).max(1.0e-3);
        let gamma_a = (W_SCALE * beta).max(W_FLOOR);

        // Emission HWHMs: blue = gamma*r, red = gamma*r*skew, skew ramped by lambda_e. The ramp folds the
        // constant divide and the (RED-BLUE) scale into one compile-time slope, so the clamp bounds become
        // [0, RED-BLUE]:  skew = SKEW_BLUE + clamp((lambda_e - SKEW_LE0)*SKEW_RAMP, 0, SKEW_RED - SKEW_BLUE).
        const SKEW_RAMP: f32 = (SKEW_RED - SKEW_BLUE) / (SKEW_LE1 - SKEW_LE0);
        let skew = SKEW_BLUE + ((lambda_e - SKEW_LE0) * SKEW_RAMP).clamp(0.0, SKEW_RED - SKEW_BLUE);
        let gamma_b = gamma_a * r;
        let gamma_r = gamma_b * skew;

        // All four inverse-CDF / normalization angles are lane-independent, so evaluate them in ONE 4-lane
        // divide + atan. The emission flank angles serve both the analytic emission normalization and
        // `sample_emission`; the excitation angles serve `excite_integral` (= gamma*excite_da) and
        // `sample_excitation` -- so neither sampler needs an atan.
        // lanes: [emis blue lower, emis red upper, excite lower, excite upper] -> (bound - center)*(1/gamma), atan.
        // ONE f32x4 reciprocal of the per-lane HWHMs serves BOTH the atan-arg scale AND the hot-path
        // inverse-HWHMs (inv_gamma_{b,r,a}), so the divide becomes a multiply and the inverses are free.
        let blue_lo = mu_a.max(EMISSION_LO);
        let bound = <S::f32x4 as GenericVector>::new([blue_lo, EMISSION_HI, EXCITATION_LO, EXCITATION_HI]);
        let center = <S::f32x4 as GenericVector>::new([lambda_e, lambda_e, mu_a, mu_a]);
        let den = <S::f32x4 as GenericVector>::new([gamma_b, gamma_r, gamma_a, gamma_a]);
        // Default-policy reciprocal is exact 1/x; it's reused (atan-arg scale below + the 3 stored
        // inverse-HWHMs), so precomputing it once is worthwhile.
        let inv = den.reciprocal();
        let [inv_gamma_b, inv_gamma_r, inv_gamma_a, _] = inv.into_array().into_array();
        let [emis_a_blo, emis_a_rhi, excite_a_lo, excite_a_hi] =
            ((bound - center) * inv).atan().into_array().into_array();

        // Derived constants (no more atans / divides on the per-eval inputs).
        let qc = Q * c;
        let blue_area = -gamma_b * emis_a_blo; // gamma_b*atan((lambda_e - blue_lo)/gamma_b)
        let red_area = gamma_r * emis_a_rhi;
        let norm = (blue_area + red_area).max(1.0e-12);
        let e_norm_rcp = 1.0 / norm;
        let emis_blue_frac = blue_area / norm;
        let excite_da = excite_a_hi - excite_a_lo;
        let excite_int = gamma_a * excite_da;
        let excite_rcp = 1.0 / excite_int.max(1.0e-12);

        Self {
            c,
            mu_a,
            gamma_a,
            inv_gamma_a,
            qc,
            lambda_e,
            gamma_b,
            gamma_r,
            inv_gamma_b,
            inv_gamma_r,
            e_norm_rcp,
            emis_a_blo,
            emis_a_rhi,
            emis_blue_frac,
            excite_a_lo,
            excite_da,
            excite_int,
            excite_rcp,
        }
    }
}

impl FluorProfile {
    // ---- profiles (SIMD over a wavelength vector) ---------------------------

    /// Peak-normalized symmetric absorption Lorentzian `$a(\lambda) = 1 / (1 + ((\lambda - \mu_a)/\gamma)^2)$`,
    /// peak 1 at `$\mu_a$`, in `[0, 1]`. Defined everywhere (including the UV); no normalization.
    #[inline(always)]
    pub fn absorption<V: NnVector>(&self, lambda: V) -> V {
        // z = (lambda - mu_a)*(1/gamma): subtract FIRST (lambda, mu_a are both ~hundreds of nm; the difference
        // is small and exact) then multiply by the precomputed 1/gamma -- folding to lambda*(1/gamma) -
        // mu_a*(1/gamma) would subtract two
        // large near-equal values (catastrophic cancellation).
        let z = (lambda - V::splat(self.mu_a)) * V::splat(self.inv_gamma_a);
        z.mul_adde(z, V::ONE).reciprocal()
    }

    /// Normalized red-skewed split-Lorentzian emission density `$e(\lambda)$` (`$\int_\text{emission} e = 1$`),
    /// peak at `$\lambda_e$`, hard Stokes truncation `$\lambda > \mu_a$` (zero below). Use as the outgoing
    /// pdf `$p(\lambda_o \mid \lambda_i)$`.
    #[inline(always)]
    pub fn emission<V: NnVector>(&self, lambda: V) -> V {
        // Per-lane flank inverse-HWHM: red (lambda >= lambda_e) vs blue. Multiply (no divide).
        let red = lambda.cmp_ge(V::splat(self.lambda_e));
        let inv_gam = red.select(V::splat(self.inv_gamma_r), V::splat(self.inv_gamma_b));
        let z = (lambda - V::splat(self.lambda_e)) * inv_gam;
        // e = e_norm_rcp/(1+z^2): the (1+z^2) reciprocal isn't reused, so a single division of the stored
        // normalization beats reciprocal-then-multiply (storing `norm` instead would need a 3rd op).
        let raw = V::splat(self.e_norm_rcp) / z.mul_adde(z, V::ONE);
        // Stokes truncation: zero where lambda <= mu_a.
        raw.zz(lambda.cmp_gt(V::splat(self.mu_a)))
    }

    /// Elastic attenuation factor `$1 - c\,a(\lambda)$` in `[0, 1]` - multiply the base reflectance
    /// `$\rho(\lambda)$` by this for the diagonal term. (The reradiation's off-diagonal fluorescent part is
    /// `$Q\,c\,a(\lambda_i)\,e(\lambda_o)$`.)
    #[inline(always)]
    pub fn elastic_attenuation<V: NnVector>(&self, lambda: V) -> V {
        self.absorption(lambda).nmul_adde(V::splat(self.c), V::ONE)
    }

    /// Rank-1 fluorescent reradiation kernel value `$Q\,c\,a(\lambda_i)\,e(\lambda_o)$` - the off-diagonal of
    /// `$M(\lambda_o, \lambda_i)$`. (Both arguments are full vectors; for the per-lane independent shift pass
    /// each lane's own `$\lambda_i$`/`$\lambda_o$`.)
    #[inline(always)]
    pub fn reradiation<V: NnVector>(&self, lambda_i: V, lambda_o: V) -> V {
        V::splat(self.qc) * self.absorption(lambda_i) * self.emission(lambda_o)
    }

    // ---- excitation-shift importance sampling (camera path: draw lambda_i) -------

    /// `$\int_\text{excitation} a(\lambda_i)\,d\lambda_i$` - the absorption integral over the UV-inclusive excitation band. This
    /// is `A` in the camera-path row weight `$\text{fluor} = Q\,c\,e(\lambda_o)\,A$`, and the normalizer of [`FluorProfile::excite_pdf`].
    #[inline(always)]
    pub fn excite_integral(&self) -> f32 {
        // excite_int = gamma*excite_da = gamma*[atan((EXCITATION_HI - mu_a)/gamma) - atan((EXCITATION_LO - mu_a)/gamma)] (precomputed).
        self.excite_int
    }

    /// Reciprocal of the excitation-pdf normalizer (`$1 / \int_\text{excitation} a$`), exposed for the integrator's row
    /// weights / MIS denominators.
    #[inline(always)]
    pub fn excite_pdf_norm(&self) -> f32 {
        self.excite_rcp
    }

    /// Importance-sample the excitation wavelength `$\lambda_i \sim a(\lambda_i)$` over the excitation band
    /// (Cauchy/Lorentzian inverse-CDF), one independent draw per lane from the uniform `$u \in [0,1)$`.
    /// Returns `$(\lambda_i, \text{pdf})$` where `$\text{pdf} = a(\lambda_i) / \int_\text{excitation} a$`. This is
    /// the per-lane shift used for the SMIS independent-shift estimator (see the module docs: multiply each
    /// lane by the product of the other lanes' returned pdfs).
    #[inline(always)]
    pub fn sample_excitation<V: NnVector>(&self, u: V) -> (V, V) {
        // z = tan(a_lo + u*da); lambda_i = mu_a + gamma*z. The a_lo/da angles are precomputed (no atan here).
        let ang = u.mul_adde(V::splat(self.excite_da), V::splat(self.excite_a_lo));
        let z = ang.tan();
        let lambda_i = z.mul_adde(V::splat(self.gamma_a), V::splat(self.mu_a));

        // pdf(lambda_i) = 1 / ((1+z^2)*int_a) -- int_a = gamma*da precomputed; one reciprocal, no divide.
        let pdf = (z.mul_adde(z, V::ONE) * V::splat(self.excite_int)).reciprocal();

        (lambda_i, pdf)
    }

    /// Excitation pdf `$p(\lambda_i) = a(\lambda_i) / \int_\text{excitation} a$` for a given `$\lambda_i$` - the
    /// MIS denominator term for strategies that did not sample this `$\lambda_i$` themselves. Zero outside the excitation band.
    #[inline(always)]
    pub fn excite_pdf<V: NnVector>(&self, lambda_i: V) -> V {
        let inside = lambda_i.cmp_ge(V::splat(EXCITATION_LO)) & lambda_i.cmp_le(V::splat(EXCITATION_HI));
        (self.absorption(lambda_i) * V::splat(self.excite_rcp)).zz(inside)
    }

    // ---- emission-shift importance sampling (light path / NEE: draw lambda_o) ----

    /// Importance-sample the emission wavelength `$\lambda_o \sim e(\lambda_o)$` over the emission band (split-Lorentzian
    /// inverse-CDF): pick a flank by area, then invert that flank's Lorentzian. Returns `$(\lambda_o, \text{pdf})$`,
    /// `$\text{pdf} = e(\lambda_o)$`. One
    /// independent draw per lane. Used light-side (sample the shift, then the emitter) and for MIS.
    #[inline(always)]
    pub fn sample_emission<V: NnVector>(&self, u: V) -> (V, V) {
        let le = V::splat(self.lambda_e);
        let gb = V::splat(self.gamma_b);
        let gr = V::splat(self.gamma_r);

        // Flank angles + split are precomputed in `new()` (lane-independent) -> no atan here.
        let a_blo = V::splat(self.emis_a_blo); // <= 0 (blue lower bound)
        let a_rhi = self.emis_a_rhi; // >= 0 (red upper bound)
        let blue_frac = V::splat(self.emis_blue_frac);

        // Blue branch: angle a_blo .. 0 as u' goes 0..1; Red branch: 0 .. a_rhi.
        let is_blue = u.cmp_lt(blue_frac);
        let u_blue = u / blue_frac; // remap [0,blue_frac)->[0,1)
        let u_red = (u - blue_frac) / (V::ONE - blue_frac); // remap [blue_frac,1)->[0,1)
        let ang_blue = u_blue.mul_adde(V::ZERO - a_blo, a_blo); // a_blo + u_blue*(0 - a_blo)
        let ang_red = u_red.scale(a_rhi);
        let ang = is_blue.select(ang_blue, ang_red);
        let gam = is_blue.select(gb, gr);
        let z = ang.tan();
        let lambda_o = z.mul_adde(gam, le);
        let pdf = self.emission(lambda_o);
        (lambda_o, pdf)
    }
}
