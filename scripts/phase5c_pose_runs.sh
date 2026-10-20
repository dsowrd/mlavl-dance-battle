#!/usr/bin/env bash
# Phase 5c: after scripts/wdsf_pose.sh: linear references and pairwise MLAVL on the appearance-free pose input
# (inner-validation epochs), then the strict evaluation. GPU 0, 4 threads per run.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
while [ "$(ls $MLAVL_DATA/features/WDSF/pose 2>/dev/null | grep -c 'npz$')" -lt 313 ] || pgrep -f "wdsf_pose.py features" >/dev/null; do sleep 120; done
for t in total Technique Execution Vocabulary Originality Musicality; do
  for feat in pose pamfn; do $MLAVL_PY ours/probe_diff.py --feat $feat --target $t --k 8; done
  $MLAVL_PY ours/probe_diff.py --feat pose --target $t --k 8 --video_only
done
run() { [ -f logs/wdsf/$1/summary.json ] || $MLAVL_PY ours/train_wdsf_pair.py --name $1 --feat pose "${@:2}" > logs/wdsf/$1.log 2>&1; }
for s in 0 1 2; do run pose_total_s$s --target total --seed $s & done
run pose_Technique_s0 --target Technique --seed 0 &
run pose_Execution_s0 --target Execution --seed 0 &
wait
E="$MLAVL_PY ours/eval_pair.py"
for t in total Technique Execution; do $E logs/wdsf/probe_pose_k8_$t; $E logs/wdsf/probe_pose_vonly_k8_$t; $E logs/wdsf/probe_pamfn_k8_$t; done
$E logs/wdsf/pose_total_s0 logs/wdsf/pose_total_s1 logs/wdsf/pose_total_s2
$E logs/wdsf/pose_Technique_s0
$E logs/wdsf/pose_Execution_s0
echo "[$(date)] done"
