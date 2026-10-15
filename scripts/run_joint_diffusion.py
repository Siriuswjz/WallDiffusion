"""Queue four joint UW diffusion heights on three independent idle GPUs."""
import datetime
import argparse
import json
import os
from pathlib import Path
from queue import Queue, Empty
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_diffusion_trial import gpu_status
from src.diffusion.data import inventory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--heights',type=int,nargs='+',default=[15,30,50,100],choices=[15,30,50,100])
    parser.add_argument('--gpus',type=int,nargs='+',default=[0,1,2])
    parser.add_argument('--resume-root',type=Path)
    args=parser.parse_args()
    # Validate all data before starting any GPU work.
    for height in args.heights:
        inventory(ROOT/'data/cf_p',height,ROOT/'data/uw_fields')
    for gpu in args.gpus:
        if not gpu_status(gpu)['idle']:raise RuntimeError(f'GPU {gpu} is occupied')
    run=args.resume_root or ROOT/'output/uw/diffusion/runs'/f'{datetime.datetime.now():%Y%m%d_%H%M%S_%f}_joint_uw_heights'
    if args.resume_root:
        if not (run/'queue_protocol.json').is_file():raise ValueError('Missing original queue protocol')
        if any((run/f'yplus_{h}').exists() for h in args.heights):
            raise ValueError('Resume may only add unstarted heights; existing results preserved')
    else:run.mkdir(parents=True,exist_ok=False)
    manifest_name=f'resume_{datetime.datetime.now():%Y%m%d_%H%M%S}.json' if args.resume_root else 'queue_protocol.json'
    (run/manifest_name).write_text(json.dumps(dict(command=[sys.executable,*sys.argv],
                  heights=args.heights,gpus=args.gpus,epochs=200,seed=42,split='70/15/15',
                  targets=['cf','p'],condition=['u','w'],observations='none'),indent=2))
    print(f'Joint run: {run}',flush=True)
    jobs=Queue()
    for height in args.heights:jobs.put(height)
    def worker(gpu):
        while True:
            try:height=jobs.get_nowait()
            except Empty:return
            # CUDA teardown can briefly retain memory after a subprocess exits.
            # Wait for release instead of dropping the next queued experiment.
            while not gpu_status(gpu)['idle']:
                print(f'GPU {gpu} not yet idle; retaining y+={height} in queue',flush=True)
                time.sleep(5)
            folder=run/f'yplus_{height}';folder.mkdir()
            command=[sys.executable,'-u',str(ROOT/'train_diffusion.py'),'--target','joint',
                     '--yplus',str(height),'--device',f'cuda:{gpu}','--seed','42','--normalization','existing',
                     '--sensor-layout','random','--density','0','--epochs','200','--prediction-type','v',
                     '--eval-interval','5','--sampling-steps','50','--batch-size','8','--workers','4',
                     '--amp','--output-root',str(folder/'training')]
            (folder/'command.json').write_text(json.dumps(command,indent=2))
            print(f'Start y+={height} on GPU {gpu}: {folder}',flush=True)
            with (folder/'console.log').open('w',buffering=1) as log:
                result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            (folder/'training_exit_code.txt').write_text(str(result.returncode))
            if result.returncode:raise RuntimeError(f'Training failed: {folder}')
            checkpoints=list((folder/'training').glob('*/best.pt'))
            if len(checkpoints)!=1:raise ValueError('Ambiguous best checkpoint')
            prediction=[sys.executable,'-u',str(ROOT/'scripts/predict_joint_diffusion.py'),
                        '--checkpoint',str(checkpoints[0]),'--output',str(folder/'test_predictions'),
                        '--device',f'cuda:{gpu}','--samples','4','--steps','50','--seed','42']
            (folder/'prediction_command.json').write_text(json.dumps(prediction,indent=2))
            with (folder/'prediction.log').open('w',buffering=1) as log:
                result=subprocess.run(prediction,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            (folder/'prediction_exit_code.txt').write_text(str(result.returncode))
            if result.returncode:raise RuntimeError(f'Prediction failed: {folder}')
            report=json.loads((folder/'test_predictions/metrics.json').read_text())
            if report['case_count']!=50 or len(list((folder/'test_predictions').glob('*.npz')))!=50:
                raise ValueError('Incomplete test predictions')
            print(f'Completed y+={height}: {report["summary"]}',flush=True)
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures=[pool.submit(worker,gpu) for gpu in args.gpus]
        for future in futures:future.result()
    print('All requested joint diffusion experiments completed',flush=True)


if __name__=='__main__':main()
