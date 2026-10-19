"""Record synchronized inference latency on one node."""
import argparse
import csv
import json
from pathlib import Path
import time


def main():
    import torch
    from tefed.engine import load_config,load_checkpoint,prepared_sample,assess_frame,write_json
    from tefed.data import load_manifest
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--node',required=True)
    p.add_argument('--client',required=True)
    p.add_argument('--requests',type=int,default=200)
    p.add_argument('--warmup',type=int,default=20)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if a.requests<200 or a.warmup<0:
        raise ValueError('Record at least 200 requests and a nonnegative warm-up')
    out=Path(a.output)
    if out.exists() or out.with_suffix('.scope.json').exists():
        raise FileExistsError('Choose a new output path for the measurement')
    cfg=load_config(a.config)
    model,router,payload=load_checkpoint(a.checkpoint,device=cfg['experiment'].get('device','cpu'))
    cfg['experiment']['method']=payload['config']['experiment']['method']
    manifest=load_manifest(cfg['experiment']['manifest'],strict_files=True)
    records=manifest.by_split('test',a.client)
    if not records:
        raise ValueError('The client has no held-out test identities')
    calibration=json.loads(Path(cfg['experiment']['calibration']).read_text(encoding='utf-8'))
    state=payload['states'][a.client]
    if payload.get('client_models'):
        model.load_state_dict(payload['client_models'][a.client])
    if payload.get('client_bn'):
        sd=model.state_dict();sd.update(payload['client_bn'][a.client]);model.load_state_dict(sd)
    expert=cfg['experiment']['method'] in {'server_moe','generic_hc','tefed'}
    device=next(model.parameters()).device
    cuda=next(model.parameters()).is_cuda
    rows=[]
    route_counts={}
    for i in range(a.warmup+a.requests):
        record=records[i%len(records)]
        if cuda:
            torch.cuda.synchronize(device)
        start=time.perf_counter()
        sample=prepared_sample(record,cfg)
        assessment=assess_frame(model,router,sample,state,calibration,cfg,experts=expert)
        if cuda:
            torch.cuda.synchronize(device)
        elapsed=(time.perf_counter()-start)*1000
        if i>=a.warmup:
            active=sorted({0,*(head for item in assessment['candidates'] for head in item['route'])})
            route_key=','.join(map(str,active))
            route_counts[route_key]=route_counts.get(route_key,0)+1
            rows.append({'node':a.node,'request':i-a.warmup,'identity':record.id,
                         'timestamp':time.time(),'latency_ms':elapsed,'active_heads':route_key,
                         'active_prediction_parameters':model.capacity(active)['active_prediction_parameters']})
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(out.with_suffix('.scope.json'),{'scope':'image loading + synchronized preprocessing + routed inference',
        'node':a.node,'device':str(next(model.parameters()).device),'requests':a.requests,'warmup':a.warmup,
        'repeated_fixed_identities':True,'state_frozen':True,'checkpoint':str(a.checkpoint),
        'capacity':model.capacity(),'router_parameters':sum(p.numel() for p in router.parameters()) if expert else 0,
        'observed_frame_head_sets':route_counts,
        'parameter_scope':'active prediction parameters count distinct tensors across all candidates in a frame; '
                          'latency includes repeated general-head modality and augmentation cue forwards'})


if __name__=='__main__':
    main()
