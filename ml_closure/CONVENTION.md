# ml_closure LAYOUT CONVENTION (in force from 2026-07-13)

Ordered by Sanaa 2026-07-13: plots first, English first, no codenames in
filenames. Every future diagnostic writes DIRECTLY into this tree.

## The tree

```
codes/           every script involved (python + SGE job scripts), with a
                 README.txt saying in one line what each does
train_models/    one folder per trained model, plain-English name;
                 inside: checkpoints -> symlink to the run dir, and
                 code_and_config_used/ = real copies of the config (with an
                 English header) and of the training code at that model's
                 commit
pngs/<full_english_name_of_diagnostic>/
                 all pngs of one diagnostic; every filename says in full
                 English what the figure is and WHICH MODEL it belongs to;
                 every folder contains <folder_name>.txt explaining, for
                 each png: what it shows, inputs, outputs, what it
                 compares, the math formula if any, and why we made it
yamls/<full_english_name_of_diagnostic_or_summary>/
                 the numbers behind the pngs; every yaml COPY carries a
                 top '# ' comment header explaining it in English
csvs_and_npz/    tables and arrays, full descriptive names; every .npz has
                 a sibling .txt listing its arrays and what each one is
```

## Rules

1. PLOTS FIRST, ENGLISH FIRST. A diagnostic is not done until its png is in
   pngs/<diagnostic>/ under a full-English name, its folder txt has an
   entry for it, and its numbers (if any) are in yamls/ or csvs_and_npz/.
2. NO CODENAMES IN FILENAMES. Model names in filenames are the plain ones
   (cylinder_steady_production, cape_baseline, cylinder_ensemble_conditioned,
   cape_ensemble_conditioned, cape_leave_out_<inlet>_fold,
   hyperparameter_grid_lr*_wd*, *_smoke_test). Run codenames
   (prod_ext150, piff_fpc_ens, ...) may appear INSIDE txt/headers in
   parentheses, never in filenames.
3. Runs still training publish partial artifacts with an `_in_training`
   suffix; the suffix is dropped when finals land.
4. Every new .npz gets its sibling .txt at creation time; every new yaml
   copy gets its English header at creation time. Not later.
5. NEVER edit originals under runs_piff/ to add headers -- header the COPY.
5a. SUBFOLDERS BY TEST CATEGORY (Sanaa order 2026-07-14, standing): inside
   every pngs/<diagnostic>/ the plots are split into subfolders named
   after the CATEGORY of test the plot shows (e.g.
   uncertainty_and_calibration/, field_panels_cylinder/,
   field_panels_cape/, inlet_reynolds_traces/), so a sigma plot or a cape
   field plot is findable without reading filenames. Each subfolder
   carries its own <subfolder_name>.txt explainer; the top-level
   <diagnostic>.txt is the index of subfolders + headlines. Reports
   (recalibration, verdicts, ...) follow the same subfolder convention.
5b. RELATIVE ERROR IN ERROR PLOTS (Sanaa order 2026-07-14, standing): when
   a figure shows a prediction error field, plot RELATIVE error —
   (pred - truth) normalized with a near-zero-safe denominator
   (|truth| + 0.01 * per-frame max|truth|), seismic, centered — so error
   is readable against heavy-tailed field amplitudes. Absolute-error
   panels may accompany, never replace.
6. LEGACY NOTE (temporary): until tonight's fleet lands, the training jobs
   still write into runs_piff/<codename>/ and the .py files physically
   live in ml_closure/ root (codes/ holds symlinks). After the last job
   lands tonight, the physical code moves into codes/ and future runs
   write straight into this tree. Everything currently in pngs/, yamls/,
   csvs_and_npz/ is a COPY; the runs_piff originals stay untouched until
   then.

## Where things came from (provenance)

Every yaml header and folder txt names its original path under runs_piff/.
The model <-> codename map is in train_models/README.txt and in each
model's README.
