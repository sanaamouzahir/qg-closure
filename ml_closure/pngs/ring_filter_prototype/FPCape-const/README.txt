FPCape-const (cape) -- ring-filter prototype figures

fields_t100.35.png
  Two rows, shared symlog scale (heavy-tailed field, eval-style linthresh =
  99th pct |Pi|). Top: raw Pi_J target | inpaint-repaired | x-notch-repaired
  | y-Nyquist-notch-repaired. Bottom: the quiescent-band column-RMS profile
  used to find the band (log-y, band shaded) | removed by inpaint | removed
  by x-notch | removed by ynotch (all raw - filtered). A removed panel
  should show ONLY the artefact column; wake structure appearing there =
  collateral damage.

cross_sections.png
  Pi vs x at three y-stations (through the wake, mid-height, near the top
  edge), raw vs the three filters, symlog y. Grey shading = the detected
  artefact band. Far from the body the raw curve oscillates inside the band
  and the filtered curves cut through it; through the wake the filter should
  hug the raw curve (real physics untouched) -- only ynotch does.

spectrum.png
  x-wavenumber power spectrum averaged over quiescent rows (|y - y_c| >
  4D), before/after each filter. The artefact = the mid/high-k bump in
  the raw curve; a good filter drops it by orders of magnitude while leaving
  low k unchanged.

Band detected automatically: columns 102..127 (width 26 px = 1.276 D),
x-station 5.645 (obstacle x_c = 5.027), artefact/background column-RMS
amplitude ratio 45.2x in the quiescent band. In-band adjacent-row
correlation (quiescent rows): +0.331 -- near -1 means the artefact is a
pure y-grid-Nyquist checkerboard and ynotch removes it surgically; far from
-1 (the cape) means part of the artefact is smooth in y and ynotch only
captures the checkerboard share.
