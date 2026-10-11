"""Summarise finished runs from logs/stdout/*.log into a markdown table for experiments/EXPERIMENTS.md.

    $MLAVL_PY scripts/summarize_runs.py logs/stdout/rg_*_1011_1548.log

Prints one row per log (best rho / MSE with epochs, wall time, max RSS) plus arithmetic and Fisher-z mean rho
and mean MSE over the given logs (use it on the four RG apparatus logs or the two Fis-V logs).
"""
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


def main(paths):
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


if __name__ == '__main__':
    main(sys.argv[1:])
