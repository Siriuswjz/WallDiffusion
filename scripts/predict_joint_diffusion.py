"""Joint UW diffusion: full-field single-sample and ensemble-mean test metrics."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.diffusion import ConditionalUNet, GaussianDiffusion
from src.diffusion.data import inventory, case_tensors
from predict_diffusion import metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--samples',type=int,default=4)
    p.add_argument('--steps',type=int,default=50)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if a.samples<1:raise ValueError('samples must be positive')
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    protocol=ck['protocol']; settings=protocol['arguments']; stats=protocol['statistics']
    if settings['target']!='joint' or protocol['architecture']['condition_channels']!=2:
        raise ValueError('Expected velocity-only joint checkpoint')
    model=ConditionalUNet(**protocol['architecture']).to(a.device).eval()
    model.load_state_dict(ck['model'])
    diffusion=GaussianDiffusion(settings['timesteps'],settings['prediction_type']).to(a.device)
    cases=inventory(settings['data_root'],settings['yplus'],settings['w_root'])
    if {k:[p.name for p in group] for k,group in cases.items()}!=protocol['cases']:
        raise ValueError('Case split differs')
    a.output.mkdir(parents=True,exist_ok=False)
    manifest=dict(command=[sys.executable,*sys.argv],checkpoint=str(a.checkpoint),epoch=ck['epoch'],
                  split='test',components=['u','w'],targets=['cf','p'],samples=a.samples,
                  steps=a.steps,seed=a.seed,device=a.device,statistics=stats,
                  cases=[p.name for p in cases['test']],observations='none',
                  uncertainty='population standard deviation across generated samples')
    (a.output/'protocol.json').write_text(json.dumps(manifest,indent=2))
    scale=np.asarray(stats['std'][2:],dtype=np.float64)[:,None,None]
    shift=np.asarray(stats['mean'][2:],dtype=np.float64)[:,None,None]
    rows=[]; sample_rows=[]
    for case in cases['test']:
        condition,clean=case_tensors(case,settings['w_root'],settings['yplus'],None,stats,0,a.seed)
        truth=clean.numpy().astype(np.float64)*scale+shift
        mean=m2=first=None
        for i in range(a.samples):
            gen=diffusion.sample(model,condition[None].to(a.device),a.steps,a.seed+i).cpu().numpy()[0]
            gen=gen.astype(np.float64)*scale+shift
            if not np.isfinite(gen).all():raise ValueError('Nonfinite sample')
            if i==0:mean=gen.copy();m2=np.zeros_like(mean);first=gen.copy()
            else:
                delta=gen-mean;mean+=delta/(i+1);m2+=delta*(gen-mean)
            for c,target in enumerate(('cf','p')):
                result=metrics(gen[c],truth[c],np.zeros(truth[c].shape,dtype=bool),np.zeros_like(truth[c]))
                result.pop('observation_rmse')
                sample_rows.append(dict(case=case.name,target=target,sample=i,**result))
        payload={'case_name':case.name}
        for c,target in enumerate(('cf','p')):
            result=metrics(mean[c],truth[c],np.zeros(truth[c].shape,dtype=bool),np.zeros_like(truth[c]))
            result.pop('observation_rmse');row=dict(case=case.name,target=target,**result)
            rows.append(row);print(json.dumps(row),flush=True)
            payload.update({f'{target}_true':truth[c].astype(np.float32),f'{target}_pred':mean[c].astype(np.float32),
                            f'{target}_pred_single':first[c].astype(np.float32),
                            f'{target}_pred_std':np.sqrt(m2[c]/a.samples).astype(np.float32)})
        np.savez_compressed(a.output/f'{case.name}.npz',**payload)
    def aggregate(records):
        return {target:{key:dict(mean=float(np.mean([r[key] for r in records if r['target']==target])),
                                std=float(np.std([r[key] for r in records if r['target']==target])))
                       for key in ('mse','relative_mse','correlation','amplitude_ratio')} for target in ('cf','p')}
    (a.output/'metrics.json').write_text(json.dumps(dict(case_count=50,ddof=0,cases=rows,
                  summary=aggregate(rows),first_sample_summary=aggregate([r for r in sample_rows if r['sample']==0]),
                  all_sample_metrics=sample_rows),indent=2))
    print('Completed 50 full-field joint test cases',flush=True)


if __name__=='__main__':main()
