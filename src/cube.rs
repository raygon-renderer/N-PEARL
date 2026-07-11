//! Trilinear lookup cube: a baked CIELAB grid of network outputs that stands in for the per-material MLP
//! forward ([`crate::DeployNet::uplift`]) -- an 8-corner fetch + lerp, cheaper when the same colours are
//! uplifted many times and the MLP cost would otherwise repeat.
//!
//! # Layout (corners, not cells)
//!
//! The grid is `N` NODES per axis (`N^3` total), each a packed `u32x8` of the 16 network outputs (two
//! halves per lane; see the `util` module). Interpolation CELLS live BETWEEN adjacent nodes (`N-1` per
//! axis), so two neighbouring cells share the four nodes on their common face -- continuity across cell
//! edges is automatic, no seams. A query floors to a base node `i` and reads `i` and `i+1` on each axis,
//! the base clamped to `N-2` so the upper corner is always in range. That clamp answers "store the lower
//! or upper corner": store NODES, address the lower one, the upper is its `+1` neighbour.
//!
//! # Reflectance interpolation (not parameter interpolation)
//!
//! A single material lookup ([`LookupCube::sample`]) returns a [`CubeSample`] that holds the 8 corner
//! parameter sets and interpolates the *resulting reflectance* per wavelength: the sigmoid sits between
//! the cached parameters and the reflectance, so interpolating the cheap-to-store parameters and
//! interpolating the reflectance are NOT the same -- and the appearance metric strongly prefers the
//! latter. [`CubeSample::reflectance`] evaluates each corner's reflectance at the wavelength vector and
//! trilinearly blends. The fluorophore parameters `(c, lambda_e, s, r)` are instead interpolated directly
//! and exposed on the sample, because the renderer's energy-conserving fluor forward samples the Stokes
//! shift from the parameters themselves, not from a baked colour.
//!
//! # Out-of-gamut nodes
//!
//! Every node simply holds the network's prediction for its Lab coordinate -- including the smooth
//! extrapolation the network produces outside the realizable gamut. Those nodes aren't reached by real
//! query cells, and where a boundary cell does touch one the network's own extrapolation blends more
//! cleanly than a hard black cliff would, so there is no sentinel and no per-corner branch.

use core::array::from_fn;

use thermite::element::float::spec::Fp16Fast;
use thermite::prelude::*;

use crate::{NnBackend, NnVector, UpliftParams};

type U16x16<B> = <B as SimdVectors>::u16x16;
type F32x16<B> = <B as SimdVectors>::f32x16;

/// How a [`LookupCube`] stores each node's 16 network outputs in memory: the per-node `Storage` type and
/// the pack/unpack between it and a live `f32x16`. Chooses the cube's space-vs-decode tradeoff.
///
/// Every node is baked once ([`LookupCube::bake`] calls [`pack`](Self::pack)) and decoded on each corner
/// fetch (`LookupCube::corner` calls [`unpack`](Self::unpack)). The two built-in choices are
/// [`F16Storage`] (half the footprint, costs a decode per fetch) and [`F32Storage`] (no decode, double the
/// footprint); see each for the measured tradeoff. Implement this trait for other formats (e.g. a different
/// quantization) by supplying a `Copy` storage type and its pack/unpack.
pub trait ParamStorage<B: NnBackend> {
    /// The in-memory representation of one node (16 network outputs). One `Vec` element per grid node, so
    /// its size sets the cube's footprint: `N^3 * size_of::<Storage>()`.
    type Storage: Copy;

    /// Encode a node's 16 outputs (lane order matches [`crate::DeployNet`]'s output) into `Storage`. Baked
    /// once per node; not on the hot path.
    fn pack(values: F32x16<B>) -> Self::Storage;
    /// Decode a node back to a live `f32x16`. Runs on every corner fetch (8 per trilinear lookup, 32 per
    /// bicubic), so its cost is the per-lookup overhead this trait trades against footprint.
    fn unpack(packed: Self::Storage) -> F32x16<B>;
}

/// Store each node as 16 IEEE-754 halves packed into a `u16x16` Half
/// the memory of [`F32Storage`], at the cost of an f16->f32 decode (~2.5 ns/node) on every corner fetch
/// _unless_ running on AVX2+F16C in which case it will be **even faster than f32** thanks
/// to better cache utilization.
pub struct F16Storage;

/// Store each node as a raw `f32x16`. No decode, double the footprint of [`F16Storage`]. The default.
pub struct F32Storage;

impl<B: NnBackend> ParamStorage<B> for F16Storage {
    type Storage = U16x16<B>;

    #[inline(always)]
    fn pack(values: F32x16<B>) -> Self::Storage {
        <U16x16<B> as PackedFloatVector<Fp16Fast, F32x16<B>>>::pack(values)
    }

    #[inline(always)]
    fn unpack(packed: Self::Storage) -> F32x16<B> {
        <U16x16<B> as PackedFloatVector<Fp16Fast, F32x16<B>>>::unpack(packed)
    }
}

impl<B: NnBackend> ParamStorage<B> for F32Storage {
    type Storage = F32x16<B>;

    #[inline(always)]
    fn pack(values: F32x16<B>) -> Self::Storage {
        values
    }

    #[inline(always)]
    fn unpack(packed: Self::Storage) -> F32x16<B> {
        packed
    }
}

/// A baked CIELAB grid of the network's 16 outputs, queried by trilinear interpolation. Generic over the
/// SIMD backend `B` (the decode/interpolation width), the node [`ParamStorage`] `S` (f32 vs f16 -- the
/// footprint/decode tradeoff, default [`F32Storage`]), and the per-axis node count `N`.
pub struct LookupCube<B: NnBackend, S: ParamStorage<B> = F32Storage, const N: usize = 64> {
    /// Row-major `N^3` grid of packed nodes: node `(ix, iy, iz)` at `(ix*N + iy)*N + iz`, holding that
    /// Lab coordinate's 16 network outputs as 16 halves. Out-of-gamut nodes are baked to BLACK.
    cell_corners: Vec<S::Storage>,
    /// Lower Lab corner of the grid AABB (E-referenced `[L, a, b]`).
    lo: [f32; 3],
    /// `(N-1) / (hi - lo)` per axis: maps a Lab coordinate into `[0, N-1]` grid space in one FMA.
    inv_step: [f32; 3],
}

/// The result of one cube lookup: the 8 surrounding corner parameter sets plus the cell-fractional
/// position, with the fluorophore parameters already interpolated. [`Self::reflectance`] interpolates
/// the reflectance per wavelength; `(c, lambda_e, s, r)` feed the renderer's fluor forward directly.
#[derive(Clone, Copy, Debug)]
pub struct CubeSample {
    /// Corner parameter sets in `(ix, iy, iz)` bit order `[000, 001, 010, 011, 100, 101, 110, 111]`.
    corners: [UpliftParams; 8],
    /// Fractional position within the cell, `[fx, fy, fz]` in `[0, 1]`.
    f: [f32; 3],
    /// Interpolated fluorophore concentration / mix.
    pub c: f32,
    /// Interpolated emission peak `lambda_e` (nm).
    pub lambda_e: f32,
    /// Interpolated Stokes shift `s` (nm).
    pub s: f32,
    /// Interpolated emission-width ratio `r`.
    pub r: f32,
}

/// Trilinear blend of 8 corner scalars in `[000..111]` order at cell-fraction `f`.
#[inline(always)]
fn tri8(v: [f32; 8], f: [f32; 3]) -> f32 {
    let l = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let (c00, c01) = (l(v[0], v[1], f[2]), l(v[2], v[3], f[2]));
    let (c10, c11) = (l(v[4], v[5], f[2]), l(v[6], v[7], f[2]));
    l(l(c00, c01, f[1]), l(c10, c11, f[1]), f[0])
}

#[thermite::dispatch(B)]
impl<B: NnBackend, S: ParamStorage<B>, const N: usize> LookupCube<B, S, N> {
    /// Bake the grid from a node generator. `f(lab)` returns the 16 network outputs for an E-referenced
    /// Lab node -- run the network at every node; out-of-gamut ones just hold its extrapolation. `lo`/`hi`
    /// are the inclusive Lab AABB the grid spans (node `i` on axis `d` sits at `lo[d] + i/(N-1)*(hi[d]-lo[d])`).
    pub fn bake(lo: [f32; 3], hi: [f32; 3], f: impl FnMut([f32; 3]) -> [f32; 16]) -> Self {
        assert!(N >= 2, "cube needs at least 2 nodes per axis");
        let step: [f32; 3] = from_fn(|d| (hi[d] - lo[d]) / (N - 1) as f32);
        let mut cell_corners = Vec::with_capacity(N * N * N);
        let mut f = f; // dispatch indirection causes this

        for ix in 0..N {
            for iy in 0..N {
                for iz in 0..N {
                    let lab = [
                        lo[0] + ix as f32 * step[0],
                        lo[1] + iy as f32 * step[1],
                        lo[2] + iz as f32 * step[2],
                    ];

                    cell_corners.push(S::pack(F32x16::<B>::new(f(lab))));
                }
            }
        }

        Self {
            cell_corners,
            lo,
            inv_step: from_fn(|d| (N - 1) as f32 / (hi[d] - lo[d])),
        }
    }

    /// Decode one node (flat index) and bound it into reconstruction parameters.
    fn corner(&self, flat: usize) -> UpliftParams {
        let o: [f32; 16] = S::unpack(self.cell_corners[flat]).into_array().into_array();
        UpliftParams::from_outputs::<B>(&o)
    }

    /// Benchmark hook: decode one node to its 16 f32 WITHOUT bounding -- isolates the f16->f32 cost from
    /// the `from_outputs` cost inside [`Self::corner`]. Not for production use.
    #[doc(hidden)]
    #[inline(always)]
    pub fn decode_raw(&self, flat: usize) -> [f32; 16] {
        S::unpack(self.cell_corners[flat]).into_array().into_array()
    }

    /// Look up an E-referenced Lab colour: fetch + bound the 8 surrounding corners and interpolate the
    /// fluorophore parameters. The returned [`CubeSample`] interpolates reflectance per wavelength on
    /// demand. Colours outside the grid AABB clamp to its boundary.
    pub fn sample(&self, lab: [f32; 3]) -> CubeSample {
        // Grid coordinate in [0, N-1]; base node clamped to N-2 so its +1 upper corner stays in range.
        let t: [f32; 3] = from_fn(|d| ((lab[d] - self.lo[d]) * self.inv_step[d]).clamp(0.0, (N - 1) as f32));
        let i0: [usize; 3] = from_fn(|d| (t[d] as usize).min(N - 2));
        let f: [f32; 3] = from_fn(|d| t[d] - i0[d] as f32);

        let row = |dx: usize, dy: usize, dz: usize| ((i0[0] + dx) * N + i0[1] + dy) * N + i0[2] + dz;
        let corners = [
            self.corner(row(0, 0, 0)),
            self.corner(row(0, 0, 1)),
            self.corner(row(0, 1, 0)),
            self.corner(row(0, 1, 1)),
            self.corner(row(1, 0, 0)),
            self.corner(row(1, 0, 1)),
            self.corner(row(1, 1, 0)),
            self.corner(row(1, 1, 1)),
        ];

        // Fluorophore params interpolate directly (the smoothness prior keeps them well-defined across
        // the inactivation boundary, so naive trilinear is correct -- needed for Stokes-shift sampling).
        CubeSample {
            c: tri8(corners.map(|p| p.c), f),
            lambda_e: tri8(corners.map(|p| p.lambda_e), f),
            s: tri8(corners.map(|p| p.s), f),
            r: tri8(corners.map(|p| p.r), f),
            corners,
            f,
        }
    }
}

impl<B: NnBackend, S: ParamStorage<B>, const N: usize> LookupCube<B, S, N> {
    /// Wrap a pre-baked node grid (e.g. decoded from a scene asset). `cell_corners` must be row-major
    /// `N^3` in [`Self::bake`] order, in the storage `S` (`S::Storage`); `lo`/`hi` its inclusive Lab AABB.
    pub fn from_packed(cell_corners: Vec<S::Storage>, lo: [f32; 3], hi: [f32; 3]) -> Self {
        assert_eq!(cell_corners.len(), N * N * N, "cube node count must be N^3");
        Self {
            cell_corners,
            lo,
            inv_step: from_fn(|d| (N - 1) as f32 / (hi[d] - lo[d])),
        }
    }
}

impl CubeSample {
    /// Evaluate each corner's reflectance and trilinearly blend with weights `w = [wx, wy, wz]`.
    #[inline(always)]
    fn blend<V: NnVector>(&self, lambda: V, w: [f32; 3]) -> V {
        let [wx, wy, wz] = w.map(V::splat);
        let c = &self.corners;
        // Blend each z-edge, then the two y-edges, then x. mix(a, b) = a*(1-t) + b*t.
        let c00 = wz.mix(c[0].reflectance(lambda), c[1].reflectance(lambda));
        let c01 = wz.mix(c[2].reflectance(lambda), c[3].reflectance(lambda));
        let c10 = wz.mix(c[4].reflectance(lambda), c[5].reflectance(lambda));
        let c11 = wz.mix(c[6].reflectance(lambda), c[7].reflectance(lambda));
        wx.mix(wy.mix(c00, c01), wy.mix(c10, c11))
    }

    /// Interpolate the base reflectance `$\rho(\lambda)$` at the wavelength vector `lambda`: evaluate each
    /// corner's reflectance and trilinearly blend. This interpolates the RESULTING reflectance (past the
    /// sigmoid), which the appearance metric prefers over blending the cached parameters. Does not include
    /// fluorescence -- that is the renderer's forward, consuming `(c, lambda_e, s, r)`.
    #[inline(always)]
    pub fn reflectance<V: NnVector>(&self, lambda: V) -> V {
        self.blend(lambda, self.f)
    }
}
