cape_band_structure -- what these figures show (plain English)

WHAT: the Pi_J training target for the CAPE runs carries a numerical artefact
band at the obstacle x-station (columns ~102-127, ~1.28 D wide, ~45x the
quiescent background). For the CYLINDER the same kind of band turned out to
be a pure row-to-row checkerboard in y (adjacent rows anti-correlated at
-0.99997), so a tiny y-Nyquist notch removes it exactly -- that filter is
approved and settled. For the cape the adjacent-row correlation is only
+0.33, so the cape artefact is NOT the same animal, and before choosing its
filter we characterize what it actually is along y and x.

HOW: we look only at "quiescent" rows -- more than 3 obstacle-heights ABOVE
the cape tip, far from any real flow -- inside the detected band, on 4
validation frames each of FPCape-const and FPCape-ramp. Whatever signal
lives there is pure artefact. (Above the tip ONLY: the cape is
bottom-attached, so the rows below y_c - 3D are near-wall wake, not
quiescent -- using them misleads both the band detector and the earlier
+0.33 row-correlation. We also split the in-band signal into a smooth
per-column "pedestal" (the column mean in y) and the fluctuation around it:
the pedestal is what pushed the prototype's raw row-correlation up to +0.33
even though the fluctuation itself is strongly checkerboard.)

FILES (one set per run, in <run>/):

yspec_autocorr.png -- six panels:
  (a) y-wavenumber spectrum of the in-band quiescent columns, raw and after
      each candidate filter. The x-axis is k_y as a fraction of the grid
      Nyquist (1.0 = the 2-pixel wave). Where the raw curve carries its
      energy IS the artefact's y-structure.
  (b) autocorrelation of the in-band signal vs row lag (0-16). A pure
      checkerboard would alternate -1,+1,-1,...; a smooth blob would decay
      slowly and stay positive.
  (c) the same autocorrelation along x INSIDE the band (is the artefact
      x-structured too?).
  (d) the column-RMS profile that localizes the band (grey = band, dashed =
      obstacle x-station, dotted vertical = any secondary packet found,
      e.g. at the outlet-sponge edge). The band is searched only within a
      few D of the obstacle -- the artefact lives at the body x-station;
      the strongest packet elsewhere is characterized separately and its
      y-structure says whether it is artefact (checkerboard) or just the
      far wake leaking above y_c + 3D (smooth).
  (e) x-wavenumber spectrum inside the band (quiescent rows).
  (f) bar chart: fraction of the in-band quiescent variance sitting at the
      exact y-Nyquist mode / near-Nyquist (top 10% of k_y) / 0.75-0.9 k_N /
      below, raw vs after each filter -- the "what fraction is checkerboard"
      answer in one picture.

fields_<filter>_t*.png -- the winning filter shown on the actual field:
  [raw target | filtered | removed (raw - filtered)], shared symlog color
  scale (the target is heavy-tailed), seismic, aspect preserved. The
  "removed" panel must show ONLY the artefact band; wake structure appearing
  there = collateral damage.

CANDIDATE FILTERS (all applied only inside the band, cosine-blended edges,
every pixel of the domain kept):
  ynotch  -- remove the exact y-Nyquist (2-pixel checkerboard) component,
             the [1,2,1]/4 smoother; the approved cylinder filter.
  ylp75   -- stronger: remove ALL y-wavelengths shorter than ~2.7 pixels
             (top 25% of k_y), spectral with a smooth taper.
  notch2d -- ylp75 plus a mild x-smoothing (Gaussian, 1.5 px) in-band.

NUMBERS: summary.yaml next to this file. The judgement numbers per filter:
"frac_band_quiescent_pi2_removed" (how much of the pure-artefact energy it
kills -- want high), "residual_band_amp_ratio" (is the band still visible
above background afterwards -- want ~1), and "frac_wake_pi2_removed" (how
much REAL wake signal it deletes -- want ~0).
