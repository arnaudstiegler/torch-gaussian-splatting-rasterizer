# gaussian-splat
Reproducing Gaussian Splatting


# TODO:
- Introduce a class for one gaussian splat
- Download some data with SFM info
    - Not sure what to do with the current binary format, which is super cumbersome, and can't be pickled without code changes
    - Ultimately, only need one scene to test things out
- get the data ingestion pipeline up
- 


# FAQ

## How is color represented for a given gaussian
It's represented using spherical harmonics: the color of a given gaussian enveloppe will differ based on the orientation.
This method is presented in: http://arxiv.org/abs/2205.14330

## What are f_dc and f_rest
f_dc is a `[N,3]` vector that corresponds to the base color in RGB format for the zero-th order spherical harmonic.
f_rest are the remaining coefficients for the harmonics where the highest-order is 3, which means 15 coefficient per color, i.e 45 coefficients total.
See: https://github.com/graphdeco-inria/gaussian-splatting/issues/485
