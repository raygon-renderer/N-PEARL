N-PEARL
=======

**Neural Predicted Emission And Reflectance from Lab**

Runtime spectral uplift for the Raygon spectral path tracer: the inference side of a
learned RGB/XYZ-to-reflectance model.

A spectral renderer integrates light transport over wavelength, so every material needs
a reflectance spectrum, but art assets are authored as RGB. Spectral uplift is the
inverse map - and because countless spectra share the same colour (metamers), the job is
not to invert a function but to pick the metamer that behaves well physically:
colour-accurate under the authoring illuminant, smooth, stable under repeated bounces,
and saturation-preserving when dimmed.

The model is trained offline; this crate only runs it:

- A small MLP (~13K parameters) maps a colour to a few reconstruction parameters.
- A fixed reconstructor builds the base reflectance from a low-degree Chebyshev series
  plus one Lorentzian resonance, through an algebraic sigmoid.
- An energy-conserving rank-1 fluorescence term reaches saturated colours that lie
  outside the reflective gamut.

Inference runs per-texel on the CPU inside the BSDF hot loop: branch-light SIMD FMA, no
transcendental functions (only `+ - * /` and `rsqrt`), with the weights shipped as a
compact quantized blob.

Documentation
-------------

The crate docs carry the full design and math (rendered with KaTeX).
