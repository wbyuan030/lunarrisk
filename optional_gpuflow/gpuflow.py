#!/usr/bin/env python3
"""LunarRisk GPU Flow: conservative local/SSH dispatcher, never buys cloud instances.
Python 3.10+, stdlib only. Jobs are user-authored shell commands.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / 'config.json'

class FlowError(RuntimeError):
    pass

def read_config(path: Path):
    data = json.loads(path.read_text(encoding='utf-8'))
    return data

def get_by_name(items, key, name):
    for item in items:
        if item[key] == name:
            return item
    raise FlowError(f'Unknown {key} {name}')

def is_allowed(node, now: dt.datetime, timezone: str = 'Asia/Shanghai'):
    policy = node.get('availability', 'always')
    if policy == 'always':
        return True
    local = now.astimezone(ZoneInfo(timezone))
    if policy == 'weekend':
        return local.weekday() >= 5
    if policy == 'disabled':
        return False
    raise FlowError(f'Unknown availability policy: {policy}')

def check_spec(job, node):
    gpus = int(job['gpus'])
    if gpus > int(node['gpus']):
        return False, f'needs {gpus} GPUs; node has {node["gpus"]}'
    if gpus and int(job.get('min_vram_gb', 0)) > int(node.get('vram_gb', 0)):
        return False, 'insufficient per-GPU VRAM'
    accelerator = job.get('accelerator', 'any')
    if gpus and accelerator not in ('any', node['accelerator']):
        return False, f'job needs {accelerator}, node is {node["accelerator"]}'
    if node['type'] not in ('local', 'ssh'):
        return False, 'unsupported node type (only local/ssh; paid cloud excluded)'
    return True, 'OK'

def exec_on(node, script, timeout=20, capture=True):
    if node['type'] == 'local':
        args = ['bash', '-lc', script]
    elif node['type'] == 'ssh':
        host = node.get('ssh_host')
        if not host or 'CHANGE_ME' in host:
            raise FlowError(f'Node {node["name"]}: configure ssh_host first')
        args = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=6', host,
                'bash -lc ' + shlex.quote(script)]
    else:
        raise FlowError('unsupported node')
    return subprocess.run(args, capture_output=capture, text=True,
                          timeout=timeout, check=False)

def probe(node, job=None):
    marker = node.get('data_ready_marker', '')
    require_data = bool(job and job.get('require_dataset', False))
    checks = ['printf "online\\n"']
    if require_data:
        if not marker:
            return False, 'dataset marker is not configured'
        checks.append('test -f '+shlex.quote(marker)+' || { echo DATA_NOT_READY; exit 10; }')
    if job and int(job['gpus']) > 0:
        if node['accelerator'] == 'cuda':
            checks.append('command -v nvidia-smi >/dev/null || { echo NVIDIA_SMI_MISSING; exit 11; }')
            checks.append('test "$(nvidia-smi -L | grep -c GPU)" -ge '+str(job['gpus'])+' || { echo GPU_COUNT_INSUFFICIENT; exit 12; }')
        elif node['accelerator'] == 'rocm':
            checks.append('command -v python3 >/dev/null || { echo PYTHON_MISSING; exit 11; }')
            checks.append('python3 -c '+shlex.quote('import torch; assert torch.version.hip is not None and torch.cuda.is_available(), "ROCm PyTorch unavailable"'))
    try:
        res = exec_on(node, '; '.join(checks), timeout=30)
        return res.returncode == 0, (res.stdout + res.stderr).strip()[:800]
    except (subprocess.TimeoutExpired, OSError, FlowError) as err:
        return False, str(err)

def choose(config, job, now, *, live=False):
    rows = []
    for node in config['nodes']:
        good, reason = check_spec(job, node)
        if good and not is_allowed(node, now, config.get('timezone', 'Asia/Shanghai')):
            good, reason = False, 'outside availability window'
        if good and live:
            good, reason = probe(node, job)
        rows.append((node, good, reason))
    priority = job.get('prefer', [])
    rows.sort(key=lambda x: (not x[1], priority.index(x[0]['name']) if x[0]['name'] in priority else 100))
    return rows

def job_cmd(job, node):
    commands = job.get('commands', {})
    cmd = commands.get(node['accelerator'], commands.get('any'))
    if not cmd:
        raise FlowError(f'Job {job["id"]} missing {node["accelerator"]} command')
    return cmd

def get_job_dir(node, job):
    # A run is unique per node/job. Use separate job IDs for seeds.
    return os.path.join(node['workdir'], 'runs', job['id'])

def start_job(node, job):
    command = job_cmd(job, node)
    directory = get_job_dir(node, job)
    q_dir, q_workdir = shlex.quote(directory), shlex.quote(node['workdir'])
    script = (f'set -e; mkdir -p {q_dir}; '
              f'if test -f {q_dir}/pid && kill -0 "$(cat {q_dir}/pid)" 2>/dev/null; then '
              f'echo ALREADY_RUNNING; exit 20; fi; '
              f'cd {q_workdir}; '
              f'nohup bash -lc {shlex.quote(command)} >{q_dir}/stdout.log 2>&1 </dev/null & '
              f'echo $! > {q_dir}/pid; '
              f'echo STARTED_PID=$(cat {q_dir}/pid)')
    result = exec_on(node, script, timeout=30)
    if result.returncode:
        raise FlowError(f'Launch failed ({result.returncode}): {(result.stderr+result.stdout)[:1000]}')
    return result.stdout.strip()

def get_status(node, job):
    directory = shlex.quote(get_job_dir(node, job))
    script = (f'if test ! -e {directory}/pid; then echo NOT_STARTED; '
              f'elif kill -0 "$(cat {directory}/pid)" 2>/dev/null; then echo PROCESS_PRESENT; '
              f'else echo PROCESS_GONE; fi; '
              f'if test -e {directory}/stdout.log; then tail -n 8 {directory}/stdout.log; fi')
    result = exec_on(node, script, timeout=15)
    if result.returncode:
        raise FlowError((result.stderr+result.stdout)[:1000])
    return result.stdout.strip()

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest='cmd', required=True)
    sub.add_parser('list')
    pp = sub.add_parser('plan'); pp.add_argument('job'); pp.add_argument('--probe', action='store_true')
    ppr = sub.add_parser('probe'); ppr.add_argument('node')
    pr = sub.add_parser('run'); pr.add_argument('job'); pr.add_argument('--node', required=True)
    pr.add_argument('--execute', action='store_true', help='launch remote/local job (no cloud purchase)')
    pr.add_argument('--yes', action='store_true', help='required for --execute')
    ps = sub.add_parser('status'); ps.add_argument('job'); ps.add_argument('--node', required=True)
    pc = sub.add_parser('cloud'); pc.add_argument('--execute-readonly', action='store_true')
    args = parser.parse_args(argv)
    config = read_config(args.config)
    now = dt.datetime.now(dt.timezone.utc)
    if args.cmd == 'list':
        print('Nodes:', *(n['name'] for n in config['nodes']), sep='\n  ')
        print('Jobs:', *(j['id'] for j in config['jobs']), sep='\n  ')
    elif args.cmd == 'plan':
        job = get_by_name(config['jobs'], 'id', args.job)
        for node, good, reason in choose(config, job, now, live=args.probe):
            print(f'{"READY" if good else "SKIP "} {node["name"]:<22} {reason}')
    elif args.cmd == 'probe':
        node = get_by_name(config['nodes'], 'name', args.node)
        print(probe(node))
    elif args.cmd == 'run':
        job = get_by_name(config['jobs'], 'id', args.job)
        node = get_by_name(config['nodes'], 'name', args.node)
        allowed = any(n['name'] == node['name'] and ok for n, ok, _ in choose(config,job,now,live=False))
        if not allowed: raise FlowError('Node not eligible by hardware/weekend policy')
        print('Node:', node['name'], '| command:', job_cmd(job,node))
        if not (args.execute and args.yes):
            print('DRY RUN. Add --execute --yes only after verifying data and GPU availability.')
            return 0
        good, reason = probe(node, job)
        if not good: raise FlowError('Preflight failed: '+reason)
        print(start_job(node,job))
    elif args.cmd == 'status':
        job = get_by_name(config['jobs'], 'id', args.job)
        node = get_by_name(config['nodes'], 'name', args.node)
        print(get_status(node, job))
    elif args.cmd == 'cloud':
        if not args.execute_readonly:
            print('Read-only CompShare query available: add --execute-readonly (never starts or purchases a VM)')
            return 0
        try:
            r = subprocess.run(['compshare','--json','instance','zones'], text=True,
                               capture_output=True,timeout=30,check=False)
        except FileNotFoundError:
            raise FlowError('Install first: pip install compshare-cli')
        if r.returncode: raise FlowError((r.stderr+r.stdout)[:1500])
        result = json.loads(r.stdout)
        print(json.dumps(result,ensure_ascii=False,indent=2)[:8000])
    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except (FlowError, ValueError, OSError, subprocess.TimeoutExpired) as exc:
        print('ERROR:', exc, file=sys.stderr)
        sys.exit(2)
