"""Wait for an idle GPU, then run one single-target diffusion experiment."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def gpu_status(index):
    result = subprocess.run(['nvidia-smi', f'--id={index}',
                             '--query-gpu=uuid,utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                            check=True, capture_output=True, text=True)
    uuid, utilization, memory = [v.strip() for v in result.stdout.strip().split(',')]
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'],
                          check=True, capture_output=True, text=True)
    processes = [line.strip() for line in apps.stdout.splitlines() if line.split(',')[0].strip() == uuid]
    idle = not processes and int(memory) <= 128 and int(utilization) <= 5
    return dict(idle=idle, utilization=int(utilization), memory_mib=int(memory), processes=processes)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--target', choices=('p','cf'), default='p')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--full', action='store_true', help='Full run: validation every 5 epochs, 50-step sampling, final 50-case test')
    p.add_argument('--no-observations', action='store_true', help='Velocity-only ablation: zero mask and zero wall observation channels')
    p.add_argument('--block-size', type=int, default=10, help='Same fixed global sensor grid as existing neural sparse models')
    p.add_argument('--wait-for-uw-run', type=Path, help='Wait for this UW training/prediction queue to finish before using its GPUs')
    p.add_argument('--poll-seconds', type=int, default=30)
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    if args.poll_seconds < 1 or args.epochs < 1 or args.block_size < 1:
        raise ValueError('epochs, block-size and poll-seconds must be positive')
    if args.check_only:
        print(json.dumps(gpu_status(args.gpu)))
        return
    condition_name = 'uw_only' if args.no_observations else f'block{args.block_size}'
    run = ROOT/('output/uw/diffusion/runs' if args.full else 'output/uw/diffusion/trials')/f'{datetime.datetime.now():%Y%m%d_%H%M%S_%f}_{args.target}_y15_{condition_name}_v'
    run.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-u', str(ROOT/'train_diffusion.py'), '--target', args.target, '--yplus', '15',
               '--device', f'cuda:{args.gpu}', '--seed', '42', '--normalization', 'existing',
               '--sensor-layout', 'random' if args.no_observations else 'grid',
               '--density', '0' if args.no_observations else str((len(set(range(0,1400,args.block_size))|{1399})*len(set(range(0,800,args.block_size))|{799}))/(1400*800)),
               '--block-size', str(args.block_size), '--epochs', str(args.epochs),
               '--prediction-type', 'v', '--eval-interval', '5' if args.full else str(args.epochs),
               '--sampling-steps', '50' if args.full else '20', '--batch-size', '8',
               '--workers', '4', '--amp', '--output-root', str(run/'training')]
    (run/'command.json').write_text(json.dumps(command, indent=2))
    (run/'queue_protocol.json').write_text(json.dumps(dict(command=[sys.executable,*sys.argv],
                    wait_for_uw_run=str(args.wait_for_uw_run.resolve()) if args.wait_for_uw_run else None),indent=2))
    print(f'Trial directory: {run}', flush=True)
    if args.wait_for_uw_run:
        queue = json.loads((args.wait_for_uw_run/'queue_protocol.json').read_text())
        print(f'Waiting for UW training/prediction queue: {args.wait_for_uw_run}',flush=True)
        with (run/'dependency.jsonl').open('a',buffering=1) as log:
            while True:
                finished=[]
                for index in queue['jobs']:
                    folders=list(args.wait_for_uw_run.glob(f'{index:02d}_*'))
                    if len(folders)!=1:continue
                    code=folders[0]/'exit_code.txt'
                    if not code.exists():continue
                    if code.read_text().strip()!='0':raise RuntimeError(f'Dependent UW training failed: {folders[0]}')
                    if queue['predict_after']:
                        prediction=Path(queue['prediction_root'])/(folders[0].name+'_uw')/'exit_code.txt'
                        if not prediction.exists():continue
                        if prediction.read_text().strip()!='0':raise RuntimeError(f'Dependent UW prediction failed: {prediction}')
                    finished.append(index)
                log.write(json.dumps(dict(time=datetime.datetime.now().isoformat(),completed=finished,total=queue['jobs']))+'\n')
                if len(finished)==len(queue['jobs']):break
                time.sleep(args.poll_seconds)
        print('Dependent UW queue completed.',flush=True)
    with (run/'queue.jsonl').open('a', buffering=1) as log:
        idle_checks = 0
        while idle_checks < 3:
            status = gpu_status(args.gpu)
            idle_checks = idle_checks+1 if status['idle'] else 0
            status.update(time=datetime.datetime.now().isoformat(), consecutive_idle_checks=idle_checks)
            log.write(json.dumps(status)+'\n')
            print(f'GPU {args.gpu}: {status}', flush=True)
            if idle_checks < 3:
                time.sleep(args.poll_seconds)
    print('GPU idle; starting diffusion training. Log:', run/'console.log', flush=True)
    with (run/'console.log').open('w', buffering=1) as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    (run/'training_exit_code.txt').write_text(str(result.returncode)+'\n')
    if result.returncode:
        print(f'Training failed with exit {result.returncode}: {run / "console.log"}', flush=True)
        sys.exit(result.returncode)
    checkpoints = list((run/'training').glob('*/best.pt'))
    if len(checkpoints) != 1:
        raise RuntimeError(f'Expected exactly one best checkpoint, found {checkpoints}')
    prediction = [sys.executable, '-u', str(ROOT/'predict_diffusion.py'), '--checkpoint', str(checkpoints[0]),
                  '--split', 'test' if args.full else 'val', '--samples', '4', '--steps', '50' if args.full else '20',
                  '--device', f'cuda:{args.gpu}', '--seed', '42',
                  '--output', str(run/('test_predictions' if args.full else 'validation_diagnostic'))]
    if not args.full:
        prediction += ['--limit', '1']
    (run/'prediction_command.json').write_text(json.dumps(prediction, indent=2))
    with (run/'prediction.log').open('w', buffering=1) as log:
        result = subprocess.run(prediction, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    (run/'prediction_exit_code.txt').write_text(str(result.returncode)+'\n')
    if result.returncode:
        sys.exit(result.returncode)
    print(f'Trial finished: {run}; best checkpoint: {checkpoints[0]}', flush=True)


if __name__ == '__main__':
    main()
