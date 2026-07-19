#!/bin/bash
# submit_ramp_sweep.sh -- sponge RAMP-WIDTH sweep (Sanaa ruling 2026-07-17:
# penalty axis exhausted; move to ramp width, colleague-endorsed).
# [fable-authored]
#
# KNOB: qg.bc.width (fraction of the domain edge). Builds the linear sponge
# ramps in Region.outlet_mask_rtd ('single'): right strip x in [1-w, 1]
# (ramp up toward the outlet) + top double strip y in [1-2w, 1] (triangle
# across the periodic image seam). Boundary sponge forcing =
# -ramp * omega / (bc.sponge * dt). width is an existing scenario key ->
# plain hydra override, NO solver change.
#
# HELD FIXED at the member values (v3 same-physics rule + this ruling):
# qg.pde.penalty (obstacle Brinkman) AND qg.bc.sponge (boundary eta coef)
# -- both passed EXPLICITLY so every run's .hydra config shows them pinned.
# NOTE (discovery 2026-07-17): the S1 penalty sweep overrode qg.pde.penalty
# only; bc.sponge stayed at the scenario value in every sweep run (verified
# in outputs/sponge_sweep/fpc/p0p7/.hydra/config.yaml: penalty 0.7, sponge
# 1.25). This script pins BOTH so the width axis is clean.
#
# Sweep values (geometry-limited; sponge must not eat the wake):
#   fpc : width0 0.025 x {1, 1.5, 2, 3, 4}    = 0.025 .. 0.1
#         (4x: right sponge from x=0.9, top strips y in [0.8,1];
#          cylinder wake 0.275 -> 0.9 ~ 12.5 D. OK.)
#   cape: width0 0.1   x {1, 1.5, 2, 2.5}     = 0.1 .. 0.25
#         (2.5x: right sponge from x=0.75, top strips y in [0.5,1];
#          cape support ends x~0.28 -> ~12 L_cape of wake. 3x would put
#          the top strips over 60% of the height -- REJECTED.)
#
# Shorts: T=35, Nx=Ny=2048, member dt/nu/mu/B/nv, const-member inlet table,
# NaN-guard armed via +qg.diag.out (phaseB_A_job.sh). Chained analyzer:
# analyze_sponge_sweep.py --param-prefix w --sense width (inlet v_rms /
# om_rms / u_deficit binding, st_leak advisory) -> rampwidth_<g>.txt.
#
# PROPOSE-GATED: do NOT run without Sanaa's GO ([QG][PROPOSE] 2026-07-17).
# Usage (on mseas, after GO, day-mode I21c):
#   ./submit_ramp_sweep.sh            # both geometries
#   ./submit_ramp_sweep.sh fpc        # one geometry
set -u

export SGE_ROOT=${SGE_ROOT:-/opt/gridengine}
export SGE_CELL=${SGE_CELL:-default}
export PATH=/opt/gridengine/bin/lx-amd64:/opt/rocks/bin:$PATH

QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
QG_DIR=$QG_ROOT/qg-simple-package-stable/src/qg
BRANCH=$QG_ROOT/qg-sgs-closure
ENS=$QG_DIR/outputs/SGS_closure_ensemble
STATE=$QG_ROOT/reporting/.reflection_pipeline
SPOOL=$QG_ROOT/reporting/pending_mail
EMAIL=${QG_NOTIFY_EMAIL:-sanaamz@mit.edu}
LOGDIR=$BRANCH/logs
mkdir -p "$LOGDIR" "$SPOOL" "$STATE"

MEMBERS_fpc="FPC-const FPC-sine FPC-ramp FPC-ou FPC-telS-A"
MEMBERS_cape="FPCape-const FPCape-sine FPCape-ramp FPCape-ou FPCape-tel"
SCEN_fpc="flow_past_cylinder_sponge"
SCEN_cape="flow_past_cape"
# 2026-07-18 Sanaa order (round-1 fpc verdict: all 5 widths FAIL, v_rms/om_rms
# 10-50x over 1e-3; v_rms improves to w=0.075 then REVERSES at 0.1): extend
# fpc x{5,6,8} = 0.125/0.15/0.2 (top double strips 25-40% of height; x8 is
# the practical ceiling). Round-1 widths skip via the done-marker. Cape NOT
# extended pending its round-1 analyzer (x3+ eats >=60% height -- decouple
# right/top widths before going bigger there).
MULTS_fpc="1 1.5 2 3 4 5 6 8"
MULTS_cape="1 1.5 2 2.5"

cfgval() {  # $1 file  $2 key -- first '  key: value' hit
    grep -E "^\s+$2:" "$1" | head -1 | awk '{print $2}'
}

for g in ${@:-fpc cape}; do
    case $g in fpc|cape) ;; *) echo "unknown geometry '$g'"; exit 1;; esac
    scen_v="SCEN_$g"; scen=${!scen_v}
    mults_v="MULTS_$g"; mults=${!mults_v}
    cm=$(eval echo \"\$MEMBERS_$g\" | awk '{print $1}')
    cfg=$ENS/$cm/config.yaml
    [ -f "$cfg" ] || cfg=$ENS/$cm/_v1_reflected/config.yaml
    tbl=$ENS/$cm/U_of_t.npz
    [ -f "$tbl" ] || tbl=$ENS/$cm/_v1_reflected/U_of_t.npz
    [ -f "$cfg" ] && [ -f "$tbl" ] || { echo "$g: const member cfg/table missing"; exit 1; }
    dtc=$(cfgval "$cfg" dt); nuv=$(cfgval "$cfg" nu)
    muv=$(cfgval "$cfg" mu); Bv=$(cfgval "$cfg" B); nvv=$(cfgval "$cfg" nv)
    penv=$(cfgval "$cfg" penalty)     # obstacle Brinkman -- HELD at member value
    spongev=$(cfgval "$cfg" sponge)   # boundary eta coef -- HELD at member value
    w0=$(cfgval "$cfg" width)         # 1x = member ramp width
    [ -n "$dtc" ] && [ -n "$nuv" ] && [ -n "$penv" ] && [ -n "$spongev" ] \
        && [ -n "$w0" ] || { echo "$g: cannot read member config values"; exit 1; }

    echo "[$g] scen=$scen width0=$w0 penalty=$penv sponge=$spongev dt=$dtc nu=$nuv"
    # cape mask is NON-CIRCULAR: ScalarRecorder requires qg.diag.length
    # (round-1 cape 1840689-92 all crashed at startup without it; value 1.0
    # = the FPCape production convention, BRANCH_LOG 2026-07-07)
    extra_diag=""
    [ "$g" = "cape" ] && extra_diag="+qg.diag.length=1.0"
    hold_ids=""; n_jobs=0
    for m in $mults; do
        w=$(awk "BEGIN{printf \"%g\", $w0 * $m}")
        wtag=${w//./p}
        rd="outputs/ramp_sweep/$g/w$wtag"
        [ -f "$QG_DIR/$rd/DNS_FR.npz" ] && { echo "  w=$w done, skip"; continue; }
        rm -rf "${QG_DIR:?}/$rd"      # stale partial
        jid=$(cd "$BRANCH" && qsub -terse -N "rw${g:0:1}_$wtag" \
            -o "$LOGDIR/" -j y -cwd -V -q ibgpu.q -l gpu=1 \
            -m a -M "$EMAIL" \
            "$BRANCH/scripts/sge/phaseB_A_job.sh" \
            scenario="$scen" qg.grid.Nx=2048 qg.grid.Ny=2048 \
            qg.time.T=35 qg.time.dt="$dtc" qg.time.save_rate=3600 \
            qg.pde.nu="$nuv" ${muv:+qg.pde.mu=$muv} ${Bv:+qg.pde.B=$Bv} \
            ${nvv:+qg.pde.nv=$nvv} \
            qg.pde.penalty="$penv" qg.bc.sponge="$spongev" \
            qg.bc.width="$w" +qg.bc.inlet_table="$tbl" \
            +qg.diag.scalar_rate=10 +qg.diag.flush_every=500 \
            +qg.diag.out="$QG_DIR/$rd/scalars.npz" $extra_diag \
            hydra.run.dir="$rd" 2>/dev/null | head -1)
        jid=${jid%%.*}
        [ -n "$jid" ] || { echo "  w=$w qsub FAILED"; exit 1; }
        echo "  w=$w -> job $jid ($rd)"
        hold_ids="$hold_ids,$jid"; n_jobs=$((n_jobs + 1))
    done
    [ -n "$hold_ids" ] || { echo "[$g] nothing to submit"; continue; }
    hold_ids=${hold_ids#,}
    n_exp=$(echo $mults | wc -w)
    aid=$(qsub -terse -N "rwAna_$g" -hold_jid "$hold_ids" -q all.q \
        -o "$LOGDIR/" -j y -cwd -V -m ea -M "$EMAIL" \
        "$BRANCH/scripts/sge/piff_tool_job.sh" analyze_sponge_sweep.py \
        --sweep-root "$QG_DIR/outputs/ramp_sweep/$g" \
        --geometry "$g" --param-prefix w --sense width \
        --n-expected "$n_exp" --t-window 25 35 \
        --out "$STATE/rampwidth_$g.txt" 2>/dev/null | head -1)
    aid=${aid%%.*}
    echo "[$g] analyzer rwAna_$g -> job $aid (pick -> $STATE/rampwidth_$g.txt)"
    { echo "To: $EMAIL"
      echo "Subject: [QG][SUBMIT][sgs-closure] ramp-width sweep fired: $g ($n_jobs widths + chained analysis)"
      echo
      echo "geometry $g | scen $scen | grid 2048x2048 | T=35 dt=$dtc | nu=$nuv mu=${muv:-.} B=${Bv:-.} nv=${nvv:-.}"
      echo "penalty=$penv bc.sponge=$spongev HELD (member values); qg.bc.width swept x{$mults} of $w0"
      echo "inlet table: $tbl | NaN-guard armed | jobs: $hold_ids | analyzer: $aid"
      echo "logs: $LOGDIR | pick file: $STATE/rampwidth_$g.txt"
      echo
      echo "NEXT: analyzer mails the metric table + pick (v_rms/om_rms/u_deficit binding, st_leak advisory)."
    } > "$SPOOL/$(date +%Y%m%dT%H%M%S)_rampsweep_$g.mail"
done
exit 0
