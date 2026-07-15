#!/bin/bash
# reflection_pipeline.sh -- agent-free stage machine for the reflection
# remediation campaign (Sanaa orders 2026-07-15, incl. "fires even if the
# session dies"). Runs from the mseas crontab every 10 min under flock.
# [fable-authored]
#
# STAGES (markers in $STATE; every transition emails via pending_mail):
#   S0 verdicts   : both reflection_probe metrics.yaml present -> parse.
#                   No geometry SUPPORTED -> email + terminal marker.
#   S1 sweep      : per SUPPORTED geometry, qsub a sponge-penalty sweep
#                   (short const-table sims, penalty 1.25..2.25 step 0.1,
#                   T=35, inlet metrics on t in [25,35]) + chained analysis
#                   (analyze_sponge_sweep.py) that writes penalty_<geom>.txt.
#   S2 reruns     : penalty picked -> archive minimal v1 provenance
#                   (LES npz + manifest -> <member>/_v1_reflected/), then
#                   resubmit EVERY member of the geometry full-length from
#                   t=0 with ONLY qg.pde.penalty changed (identical configs;
#                   simpler-correct vs const-branching -- no IC plumbing).
#                   Each sim chains an inlet-check job (same analyzer,
#                   --check mode) appending PASS/FAIL to member_checks.txt.
#   S3 rebuild    : ALL members PASS -> fire the gaussian target rebuild
#                   (existing gaussian_rebuild_job.sh) -- plain gaussian,
#                   NO ylp75 (Sanaa: no additional filter).
#   S4 artifacts  : rebuild done -> fire the standing artifact suite; a FAIL
#                   (checkerboard back) -> FLAG email + HOLD (ylp75 decision
#                   is Sanaa's).
#   S5 retrain    : artifacts clean -> fire piff retrains on the clean data
#                   (conf_piff_<g>_gjs_clean.yaml, same masks/loss) + I18
#                   monitor units.
# Any stage whose tool/preflight is missing: ONE [QG][FLAG] email, pipeline
# HOLDS at that stage (marker flag_<stage>), everything stays armed. Drop the
# named file it asks for (or fix the tool) and the next tick resumes. The
# machine never guesses and never edits code (I24 philosophy).
set -u
export PATH=/opt/rocks/bin:$PATH
QG_ROOT=/gdata/projects/ml_scope/Closure_modeling/QG-closure
QG_DIR=$QG_ROOT/qg-simple-package-stable/src/qg
BRANCH=$QG_ROOT/qg-sgs-closure
ML=$BRANCH/ml_closure
ENS=$QG_DIR/outputs/SGS_closure_ensemble
STATE=$QG_ROOT/reporting/.reflection_pipeline
SPOOL=$QG_ROOT/reporting/pending_mail
EMAIL=${QG_NOTIFY_EMAIL:-sanaamz@mit.edu}
LOGDIR=$BRANCH/logs
mkdir -p "$STATE" "$SPOOL" "$LOGDIR"

mail_once() {  # $1 marker  $2 subject  $3 body
    [ -e "$STATE/mailed_$1" ] && return 0
    { echo "To: $EMAIL"; echo "Subject: $2"; echo; printf '%b\n' "$3"; } \
        > "$SPOOL/$(date +%Y%m%dT%H%M%S)_refl_$1.mail"
    touch "$STATE/mailed_$1"
}
flag_hold() {  # $1 stage  $2 body -- FLAG once and hold the stage
    mail_once "flag_$1" "[QG][FLAG][sgs-closure] reflection pipeline HOLDING at $1" \
        "$2\n\nPipeline stays armed; fix and it resumes on the next 10-min tick."
    touch "$STATE/flag_$1"
}

MEMBERS_fpc="FPC-const FPC-sine FPC-ramp FPC-ou FPC-telS-A"
MEMBERS_cape="FPCape-const FPCape-sine FPCape-ramp FPCape-ou FPCape-tel"
SCEN_fpc="flow_past_cylinder_sponge"
SCEN_cape="flow_past_cape_sponge"
PENALTIES="1.25 1.35 1.45 1.55 1.65 1.75 1.85 1.95 2.05 2.15 2.25"

# ---------------------------------------------------------------- S0
[ -e "$STATE/terminal" ] && exit 0
if [ ! -e "$STATE/s0_done" ]; then
    supported=""
    for g in fpc cape; do
        m=$ML/runs_piff/piff_${g}_gjs_ylp75/reflection_probe/*/metrics.yaml
        # shellcheck disable=SC2086
        f=$(ls $m 2>/dev/null | head -1)
        [ -n "$f" ] || exit 0            # probes not done yet -- wait
        if grep -q "REFLECTION-SUPPORTED" "$f"; then supported="$supported $g"; fi
    done
    echo "$supported" > "$STATE/supported_geoms"
    touch "$STATE/s0_done"
    if [ -z "${supported// /}" ]; then
        touch "$STATE/terminal"
        mail_once s0_exon "[QG][MONITOR][sgs-closure] reflection pipeline: EXONERATED both geometries -- pipeline ends" \
            "Both probes returned REFLECTION-EXONERATED. No sponge campaign fires.\nNEXT: read the filter-variant control (test C) in reports/reflection_probe_*/ to pick target-side vs model-side."
        exit 0
    fi
    mail_once s0_sup "[QG][MONITOR][sgs-closure] reflection pipeline: SUPPORTED ->$supported -- firing sponge sweeps" \
        "Probe verdict REFLECTION-SUPPORTED for:$supported.\nS1 sponge sweeps (penalty $PENALTIES, T=35 const-table shorts) queue on the GPU now.\nPer-stage emails follow; pipeline is crontab-resident (survives sessions)."
fi

# ---------------------------------------------------------------- S1
for g in $(cat "$STATE/supported_geoms" 2>/dev/null); do
    scen_var="SCEN_$g"; scen=${!scen_var}
    if [ ! -e "$STATE/s1_fired_$g" ]; then
        # preflight: scenario conf + worker + analyzer + a const inlet table
        [ -f "$QG_DIR/conf/scenario/$scen.yaml" ] || { flag_hold "s1_$g" "scenario conf $scen.yaml missing under $QG_DIR/conf/scenario/"; continue; }
        [ -f "$ML/analyze_sponge_sweep.py" ] || { flag_hold "s1_$g" "analyze_sponge_sweep.py missing in ml_closure"; continue; }
        [ -f "$BRANCH/scripts/sge/phaseB_A_job.sh" ] || { flag_hold "s1_$g" "phaseB_A_job.sh worker missing"; continue; }
        # shared tables/ dir is MISSING post-yield (2026-07-15 incident) --
        # use the const member's own manifest table (U(t<30)=2.0 verified;
        # per-member S2 checks catch any higher-U insufficiency later)
        cm=$(eval echo \"\$MEMBERS_$g\" | awk '{print $1}')
        tbl=$ENS/$cm/U_of_t.npz
        [ -f "$tbl" ] || { flag_hold "s1_$g" "member table $tbl missing"; continue; }
        hold_ids=""
        for p in $PENALTIES; do
            rd="outputs/sponge_sweep/$g/p${p/./p}"
            jid=$(cd "$BRANCH" && qsub -terse -N "sw${g:0:1}_${p/./}" \
                -o "$LOGDIR/" -j y -cwd -V -q ibgpu.q -l gpu=1 \
                -m a -M "$EMAIL" \
                "$BRANCH/scripts/sge/phaseB_A_job.sh" \
                scenario="$scen" qg.grid.Nx=2048 qg.grid.Ny=2048 \
                qg.time.T=35 qg.time.save_rate=3600 qg.pde.penalty="$p" \
                +qg.bc.inlet_table="$tbl" \
                hydra.run.dir="$rd" 2>/dev/null | head -1)
            jid=${jid%%.*}; hold_ids="$hold_ids,$jid"
        done
        hold_ids=${hold_ids#,}
        qsub -terse -N "swAna_$g" -hold_jid "$hold_ids" -q all.q \
            -o "$LOGDIR/" -j y -cwd -V -m ea -M "$EMAIL" \
            "$BRANCH/scripts/sge/piff_tool_job.sh" analyze_sponge_sweep.py \
            --sweep-root "$QG_DIR/outputs/sponge_sweep/$g" \
            --geometry "$g" --t-window 25 35 \
            --out "$STATE/penalty_$g.txt" >/dev/null 2>&1
        touch "$STATE/s1_fired_$g"
        mail_once "s1_$g" "[QG][SUBMIT][sgs-closure] sponge sweep fired: $g (11 penalties + chained analysis)" \
            "jobs sw${g:0:1}_* on ibgpu.q, analysis swAna_$g held on all of them.\nPick lands in $STATE/penalty_$g.txt; S2 reruns fire automatically."
    fi

    # ------------------------------------------------------------ S2
    if [ -f "$STATE/penalty_$g.txt" ] && [ ! -e "$STATE/s2_fired_$g" ]; then
        P=$(head -1 "$STATE/penalty_$g.txt")
        case $P in ''|*[!0-9.]*) flag_hold "s2_$g" "penalty_$g.txt unparsable: '$P'"; continue;; esac
        mem_var="MEMBERS_$g"
        for m in ${!mem_var}; do
            md=$ENS/$m
            [ -d "$md" ] || { flag_hold "s2_$g" "member dir $md missing"; continue 2; }
            # middle-ground archive (Sanaa-approved): LES npz + manifests kept,
            # DNS_FR overwritten by the rerun
            arch=$md/_v1_reflected
            if [ ! -d "$arch" ]; then
                mkdir -p "$arch"
                mv "$md"/DNS_LES_s4*.npz "$arch/" 2>/dev/null
                cp "$md/DATASET_MANIFEST.md" "$arch/" 2>/dev/null
                cp "$md/config.yaml" "$arch/" 2>/dev/null
                cp "$md/U_of_t.npz" "$arch/" 2>/dev/null
            fi
            # member-local manifest table (shared tables/ dir lost post-yield);
            # the ARCHIVED copy is passed so the rerun can never clobber it
            tbl=$arch/U_of_t.npz
            [ -f "$tbl" ] || { flag_hold "s2_$g" "U_of_t.npz missing for $m (member dir and archive)"; continue 2; }
            dtov=$(grep -E '^\s+dt:' "$arch/config.yaml" | head -1 | awk '{print $2}')
            scen_var2="SCEN_$g"
            sid=$(qsub -terse -N "rr_${m:0:9}" -o "$LOGDIR/" -j y -cwd -V \
                -q ibgpu.q -l gpu=1 -m ea -M "$EMAIL" \
                "$BRANCH/scripts/sge/phaseB_A_job.sh" \
                scenario="${!scen_var2}" qg.grid.Nx=2048 qg.grid.Ny=2048 \
                qg.time.save_rate=3600 ${dtov:+qg.time.dt=$dtov} \
                qg.pde.penalty="$P" +qg.bc.inlet_table="$tbl" \
                hydra.run.dir="outputs/SGS_closure_ensemble/$m" 2>/dev/null | head -1)
            sid=${sid%%.*}
            qsub -terse -N "ck_${m:0:9}" -hold_jid "$sid" -q all.q \
                -o "$LOGDIR/" -j y -cwd -V -m a -M "$EMAIL" \
                "$BRANCH/scripts/sge/piff_tool_job.sh" analyze_sponge_sweep.py \
                --check-run "$md" --geometry "$g" --member "$m" \
                --append "$STATE/member_checks_$g.txt" >/dev/null 2>&1
        done
        touch "$STATE/s2_fired_$g"
        mail_once "s2_$g" "[QG][SUBMIT][sgs-closure] reruns fired: $g at penalty $P (v1 LES archived to _v1_reflected/)" \
            "All ${g} members resubmitted full-length, ONLY qg.pde.penalty=$P changed.\nEach sim chains an inlet-check appending to member_checks_$g.txt; S3 fires when ALL PASS."
    fi

    # ------------------------------------------------------------ S3-S5
    mem_var="MEMBERS_$g"
    n_mem=$(echo ${!mem_var} | wc -w)
    if [ -f "$STATE/member_checks_$g.txt" ] && [ ! -e "$STATE/s3_fired_$g" ]; then
        P=$(head -1 "$STATE/penalty_$g.txt" 2>/dev/null || echo '?')
        n_pass=$(grep -c PASS "$STATE/member_checks_$g.txt" 2>/dev/null || echo 0)
        n_fail=$(grep -c FAIL "$STATE/member_checks_$g.txt" 2>/dev/null || echo 0)
        if [ "$n_fail" -gt 0 ]; then
            flag_hold "s3_$g" "inlet check FAILED for:\n$(grep FAIL "$STATE/member_checks_$g.txt")\nSponge $P insufficient for these members -- needs a ruling (higher penalty / wider ramp)."
        elif [ "$n_pass" -eq "$n_mem" ]; then
            if [ -f "$BRANCH/scripts/sge/gaussian_rebuild_job.sh" ]; then
                qsub -terse -N "gre_$g" -o "$LOGDIR/" -j y -cwd -V -q ibgpu.q -l gpu=1 \
                    -m ea -M "$EMAIL" "$BRANCH/scripts/sge/gaussian_rebuild_job.sh" \
                    >/dev/null 2>&1 \
                    && touch "$STATE/s3_fired_$g" \
                    && mail_once "s3_$g" "[QG][SUBMIT][sgs-closure] $g reruns ALL PASS -- gaussian target rebuild fired (plain gaussian, NO ylp75)" \
                       "member_checks_$g.txt: $n_pass/$n_mem PASS.\nAfter rebuild: artifact suite; if the y-Nyquist checkerboard reappears the pipeline FLAGS and holds for your ylp75 ruling."
            else
                flag_hold "s3_$g" "gaussian_rebuild_job.sh not found -- confirm the rebuild invocation for the rerun fields"
            fi
        fi
    fi
done
exit 0
