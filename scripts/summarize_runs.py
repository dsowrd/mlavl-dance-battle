"""Summarise finished runs from logs/stdout/*.log into a markdown table for experiments/EXPERIMENTS.md.

    $MLAVL_PY scripts/summarize_runs.py logs/stdout/rg_*_1011_1548.log

Prints one row per log (best rho / MSE with epochs, wall time, max RSS) plus arithmetic and Fisher-z mean rho
and mean MSE over the given logs (use it on the four RG apparatus logs or the two Fis-V logs).

    $MLAVL_PY scripts/summarize_runs.py --seeds logs/stdout/rg_*_1011_1548.log logs/stdout/rg_*_s[12]_*.log

With --seeds the logs are grouped by task (the apparatus / score in the file name; the seed suffix _sN and the
timestamp are ignored) and a second table gives mean +- std over seeds per task, plus the mean over tasks.
"""
import statistics
import math
import re
import sys


def parse(path):
    txt = open(path, errors='replace').read()
    r = {'log': path.split('/')[-1]}
    m = re.search(r'commit (\w+) gpu (\d+)', txt)
    if m:
        r['commit'], r['gpu'] = m.group(1), m.group(2)
    m = re.search(r'Best Test Coef: ([\d.-]+)\tBest Test Coef_Mse: ([\d.]+)\tBest Test Epoch: (\d+)\tBest Test Mse: ([\d.]+)'
                  r'\tBest Test Mse_Coef: ([\d.-]+)\tBest Mse Epoch: (\d+)', txt)
    if m:
        r.update(rho=float(m.group(1)), rho_mse=float(m.group(2)), rho_ep=int(m.group(3)),
                 mse=float(m.group(4)), mse_rho=float(m.group(5)), mse_ep=int(m.group(6)))
    m = re.search(r'Elapsed \(wall clock\) time.*?: ([\d:.]+)', txt)
    if m:
        r['wall'] = m.group(1)
    m = re.search(r'Maximum resident set size \(kbytes\): (\d+)', txt)
    if m:
        r['rss_gb'] = int(m.group(1)) / 2**20
    r['epochs'] = len(re.findall(r'^Epoch: ', txt, flags=re.M))
    r['crashed'] = bool(re.search(r'Traceback|Killed|out of memory', txt))
    return r


def task_of(log):
    # rg_Ball_1011_1548.log -> ('RG', 'Ball'); rg_Ball_s1_1011_1736.log -> ('RG', 'Ball'); fisv_TES_s2_... -> ('Fis-V', 'TES')
    parts = log[:-4].split('_')
    return {'rg': 'RG', 'fisv': 'Fis-V'}.get(parts[0], parts[0]), parts[1]


def seed_of(log):
    m = re.search(r'_s(\d+)_', log)
    return int(m.group(1)) if m else 0


def ms(xs, fmt):
    return (f'{statistics.mean(xs):{fmt}} ± {statistics.stdev(xs):{fmt}}' if len(xs) > 1 else f'{xs[0]:{fmt}}') + f' (n={len(xs)})'


def seed_table(rows):
    groups = {}
    for r in rows:
        if 'rho' in r:
            groups.setdefault(task_of(r['log']), []).append(r)
    print('\n| task | seeds | best ρ mean ± std | MSE @best ρ | best MSE mean ± std | ρ @best MSE |')
    print('| --- | --- | --- | --- | --- | --- |')
    per_task = {}
    for (ds, task), rs in sorted(groups.items()):
        rs.sort(key=lambda r: seed_of(r['log']))
        seeds = ','.join(str(seed_of(r['log'])) for r in rs)
        rho, rmse, mse, mrho = ([r[k] for r in rs] for k in ('rho', 'rho_mse', 'mse', 'mse_rho'))
        per_task[(ds, task)] = (statistics.mean(rho), statistics.mean(rmse), statistics.mean(mse), statistics.mean(mrho))
        print(f'| {ds} {task} | {seeds} | {ms(rho, ".3f")} | {ms(rmse, ".2f")} | {ms(mse, ".2f")} | {ms(mrho, ".3f")} |')
    for ds in sorted({d for d, _ in per_task}):
        v = [per_task[k] for k in per_task if k[0] == ds]
        if len(v) > 1:
            rho = [x[0] for x in v]
            z = math.tanh(sum(math.atanh(min(x, 0.999999)) for x in rho) / len(rho))
            print(f'| **{ds} avg over {len(v)} tasks** (seed-mean per task) | | **{sum(rho) / len(rho):.3f}** (Fisher z {z:.3f}) | '
                  f'{sum(x[1] for x in v) / len(v):.2f} | **{sum(x[2] for x in v) / len(v):.2f}** | {sum(x[3] for x in v) / len(v):.3f} |')


def main(paths):
    seeds = '--seeds' in paths
    paths = [p for p in paths if p != '--seeds']
    rows = [parse(p) for p in paths]
    print('| log | commit | GPU | epochs | wall | max RSS | best ρ (ep) | MSE @best ρ | best MSE (ep) | ρ @best MSE |')
    print('| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |')
    for r in rows:
        if 'rho' not in r:
            print(f"| {r['log']} | {r.get('commit', '?')} | {r.get('gpu', '?')} | {r['epochs']} | "
                  f"{'CRASHED' if r['crashed'] else 'running'} | | | | | |")
            continue
        print(f"| {r['log']} | {r.get('commit', '?')} | {r.get('gpu', '?')} | {r['epochs']} | {r.get('wall', '?')} | "
              f"{r.get('rss_gb', float('nan')):.1f} GB | {r['rho']:.3f} ({r['rho_ep']}) | {r['rho_mse']:.2f} | "
              f"{r['mse']:.2f} ({r['mse_ep']}) | {r['mse_rho']:.3f} |")
    done = [r for r in rows if 'rho' in r]
    if len(done) > 1:
        rhos = [r['rho'] for r in done]
        z = [math.atanh(min(x, 0.999999)) for x in rhos]
        print(f"\nmean ρ (arithmetic) = {sum(rhos) / len(rhos):.3f}; "
              f"mean ρ (Fisher z) = {math.tanh(sum(z) / len(z)):.3f}; "
              f"mean MSE @best ρ = {sum(r['rho_mse'] for r in done) / len(done):.2f}; "
              f"mean best MSE = {sum(r['mse'] for r in done) / len(done):.2f}")
    if seeds:
        seed_table(rows)


if __name__ == '__main__':
    main(sys.argv[1:])
