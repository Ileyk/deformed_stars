# deformed_stars

## Trace a closed contour

`trace_contour.py` traces one closed level set using adaptive RK45 integration,
then resamples it at approximately uniform arc-length intervals. Set `Q` and
`N_POINTS` at the top of the script, then run:

```sh
python trace_contour.py
```

The field is
`f(x, y) = -q/(1+q)/sqrt(x^2+y^2) - 1/(1+q)/sqrt(x^2+(y-1)^2) - 0.5*(x^2+(y-yG)^2)`,
where `yG = 1/(1+q)`. `Q` must be finite and positive. The field is strictly
negative. The script finds the collinear Lagrange points in the order L3
(`y < 0`), L1 (`0 < y < 1`), and L2 (`y > 1`), then automatically sets
`F0 = f(L1)`. Their coordinates and the resulting `F0` are printed when the
script runs.

Set `SEED_X_RANGE` and `SEED_Y_MIN` to define the seed search region. Its
upper y bound is set automatically to `0.99 * y(L1)`. The search uses a
`SEED_SEARCH_GRID_POINTS` square grid and refines a detected crossing; if it
finds no crossing in that box, it raises an error.

The final contour is restricted to the `x >= 0` half and resampled from L1 to
the next crossing of `x = 0`. The script writes `contour_points.csv` with
`x,y,g_eff` columns, where `g_eff = |grad(f)|` at each sampled point. It
requires NumPy, SciPy, and Matplotlib. It also displays and saves
`contour_map.png`: the
selected level is drawn as a thick black line over 20 discrete color bands,
spanning magnitudes from 0.1 to 10 times `abs(F0)` on a logarithmic scale;
colorbar labels show the corresponding negative field values. Cyan circles
mark the adaptive RK45 points, and red crosses mark the uniformly spaced
points written to the CSV, which begins at L1; a yellow star marks the RK45
seed and a green diamond marks L1. If no seed is found, the map is still
displayed and saved without the point markers. Its axes are always
`x=[-1, 1]` and `y=[-1, 2]`.

## To do

1. Represent the relative change of temperature T/T_{MAX} along the 1D contour with colored markers, for convective (`β = 0.08`) and radiative (`β = 0.25`) envelopes, where `β` is the exponent relating temperature to the local gravity field in von Zeipel theorem..
2. Include the Roche lobe filling factor (<1) as an additional degree-of-freedom, to allow the exploration of sub-RLOF photospheres.  
3. Compute the local normal vector.
3. Compute the local relative flux vector at any point in space beyond the
   photosphere.
4. Refine the curve sampling around the L1 point.
5. Extend the model to 3D and introduce 2D patches (e.g. w/ Voronoi from points?) refined around the L1 point.
6. Scale the temperature using the total stellar luminosity and derive the
   absolute temperature map on the photosphere.
7. Derive the absolute flux map at any point beyond the photosphere: a tidally
   deformed version of the finite-disk CAK correction.
