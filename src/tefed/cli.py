"""Command-line entry points for preview, inference, training and evaluation."""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

from . import __version__


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def parser():
    p=argparse.ArgumentParser(prog='tefed',description='TE-Fed input preview, inference, training and evaluation')
    p.add_argument('--version',action='version',version=f'%(prog)s {__version__}')
    commands=p.add_subparsers(dest='command',required=True)
    commands.add_parser('methods',help='List implemented comparison strategies')
    preview=commands.add_parser('preview',help='Preview RGB, depth and NIR inputs without a checkpoint')
    preview.add_argument('--example',required=True,help='Example directory containing manifest.json and original PNGs')
    preview.add_argument('--output',required=True,help='New directory for preview images and display metadata')
    infer=commands.add_parser('infer',help='Run frozen single-sample inference with a compatible TE-Fed checkpoint')
    infer.add_argument('--checkpoint',required=True)
    infer.add_argument('--calibration',required=True,help='Frozen calibration JSON associated with the checkpoint')
    infer.add_argument('--rgb',required=True)
    infer.add_argument('--depth',required=True)
    infer.add_argument('--nir',required=True,help='Prepared uint8 PNG or floating-point [0,1] NPY; not the preview display image')
    infer.add_argument('--sample-id',required=True)
    infer.add_argument('--client-id',required=True,help='Client whose frozen state exists in the checkpoint')
    infer.add_argument('--output',required=True,help='New directory for predictions and the RGB overlay')
    infer.add_argument('--device',default='cpu')
    infer.add_argument('--display-confidence',type=float,default=.25,help='Overlay threshold only; all model candidates remain in predictions.json')
    infer.add_argument('--class-names',help='Optional JSON mapping of class IDs to names')
    infer.add_argument('--round-id',type=int,default=0,help='Reference-augmentation seed context; existing evaluation defaults to 0')
    validate=commands.add_parser('validate',help='Check manifests and partition/calibration identities')
    validate.add_argument('manifest')
    validate.add_argument('--check-files',action='store_true')
    for name in ['pretrain','train']:
        q=commands.add_parser(name)
        q.add_argument('--config',required=True)
        q.add_argument('--seed',type=int,default=0)
        q.add_argument('--output',required=True)
        q.add_argument('--method')
        q.add_argument('--rounds',type=positive_integer)
        q.add_argument('--resume',help='Continue from a committed round snapshot in the same output directory')
    cal=commands.add_parser('calibrate')
    cal.add_argument('--config',required=True)
    cal.add_argument('--checkpoint',required=True)
    cal.add_argument('--output',required=True)
    cal.add_argument('--seed',type=int,default=0)
    cal.add_argument('--strict-count',type=int,default=15)
    evaluate=commands.add_parser('evaluate')
    evaluate.add_argument('--checkpoint',required=True)
    evaluate.add_argument('--config',required=True)
    evaluate.add_argument('--output',required=True)
    smoke=commands.add_parser('smoke',help='Run software smoke checks')
    smoke.add_argument('--output',required=True)
    smoke.add_argument('--device',default='cpu',help='PyTorch device, e.g. cpu or cuda:0')
    smoke.add_argument('--amp',action='store_true',help='Enable CUDA mixed precision')
    return p


def _calibrate(args):
    import numpy as np
    import torch
    from .engine import load_config,load_checkpoint,write_json,prepared_sample,assess_frame,apply_input_policy
    from .model import sample_tensor
    from .losses import WeightedNativeLoss
    from .data import load_manifest
    from .calibration import build_calibration
    config=load_config(args.config)
    device=config['experiment'].get('device','cpu')
    model,router,payload=load_checkpoint(args.checkpoint,device=device)
    model.eval(); router.eval()
    criterion=WeightedNativeLoss(model,0)
    manifest=load_manifest(config['experiment']['manifest'],strict_files=True)
    def prep(sample):
        return apply_input_policy(sample,config)
    @torch.no_grad()
    def loss(sample):
        sample=prep(sample)
        x=sample_tensor(sample,device,rgb_only=config['model'].get('detector_rgb_only',False))
        return float(criterion(model(x,(0,))[0],sample['targets'],[1.]*len(sample['targets']),device)[0])
    @torch.no_grad()
    def predict(sample):
        sample=prep(sample)
        x=sample_tensor(sample,device,rgb_only=config['model'].get('detector_rgb_only',False))
        return model.predict_heads(x,(0,),confidence=config.get('evaluation',{}).get('prediction_confidence',.01),temperature=1.)[0][0]
    artifact=build_calibration(manifest,loss,predict,image_size=config['model'].get('image_size',640),
                               seed=args.seed,task=config.get('task','tomato'),strict_count=args.strict_count,
                               progress=lambda *parts,**kwargs:print(*parts,flush=True))
    # Compute Generic HC priors from observed validation fields.
    all_attrs=[]
    prior={'q':.5,'p':.5,'h':[.2]*5}
    general_config=deepcopy(config)
    for record in manifest.records:
        if record.id not in manifest.calibration_ids:
            continue
        assessed=assess_frame(model,router,prepared_sample(record,config),prior,artifact,general_config,experts=False)
        all_attrs.extend(c['attrs'] for c in assessed['candidates'])
    if all_attrs:
        attrs=[]; availability=[]
        for i in range(8):
            values=[a[i] for a in all_attrs if a[i] is not None]
            attrs.append(float(np.median(values)) if values else 0.)
            availability.append(int(bool(values)))
        artifact['generic_priors']={name:{'attrs':attrs,'availability':availability}
                                    for name in ['illum','occ','ctx','mat','sens']}
        artifact['generic_prior_protocol']='Validation medians shared across coarse condition families'
    artifact['initial_checkpoint_sha256']=__import__('hashlib').sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    write_json(args.output,artifact)
    print(f'Wrote frozen validation artifact: {args.output}')


def main(argv=None):
    args=parser().parse_args(argv)
    if args.command=='preview':
        from .preview import preview_example
        try:
            result=preview_example(args.example,args.output)
        except (OSError,ValueError,TypeError,KeyError) as error:
            parser().exit(2,f'Preview failed: {error}\n')
        print(json.dumps({'output':str(Path(args.output).resolve()),'images':result['outputs'],
                          'inference_executed':False},ensure_ascii=False,indent=2))
        return 0
    if args.command=='infer':
        try:
            from .inference import run_inference
        except ModuleNotFoundError as error:
            parser().exit(2,f'Inference dependencies are unavailable ({error}). Run uv sync --locked --extra cpu.\n')
        try:
            result=run_inference(checkpoint=args.checkpoint,calibration=args.calibration,
                rgb=args.rgb,depth=args.depth,nir=args.nir,sample_id=args.sample_id,
                client_id=args.client_id,output=args.output,device=args.device,
                display_confidence=args.display_confidence,class_names=args.class_names,
                round_id=args.round_id)
        except (OSError,ValueError,TypeError,KeyError,RuntimeError) as error:
            parser().exit(2,f'Inference failed: {error}\n')
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0
    if args.command=='methods':
        from .baselines import METHOD_REGISTRY
        for key,description in METHOD_REGISTRY.items():
            print(f'{key}: {description}')
        return 0
    if args.command=='validate':
        from .data import load_manifest
        manifest=load_manifest(args.manifest,strict_files=args.check_files)
        print(json.dumps({'records':len(manifest.records),'calibration_count':len(manifest.calibration_ids),
                          'partitions':{key:len(manifest.by_split(key)) for key in ['labeled','unlabeled','validation','test','hard']}},indent=2))
        return 0
    if args.command=='calibrate':
        _calibrate(args)
        return 0
    from .engine import load_config,run_experiment,pretrain,load_checkpoint,evaluate_model,write_json
    if args.command in {'pretrain','train'}:
        config=load_config(args.config)
        if args.method:
            config['experiment']['method']=args.method
        if args.rounds is not None:
            config['experiment']['rounds']=args.rounds
        resume_options={'resume':args.resume} if args.resume else {}
        result=(pretrain if args.command=='pretrain' else run_experiment)(config,seed=args.seed,output_dir=args.output,**resume_options)
        print(json.dumps({'checkpoint':result['checkpoint'],'output_dir':result['output_dir']},indent=2))
        return 0
    if args.command=='evaluate':
        import hashlib
        from .data import load_manifest
        from .engine import evaluate_remote_models,validate_run_config
        config=load_config(args.config)
        validate_run_config(config)
        model,router,payload=load_checkpoint(args.checkpoint,device=config['experiment'].get('device','cpu'))
        # Inference uses training method/model metadata from the checkpoint.
        config['experiment']['method']=payload['config']['experiment']['method']
        config['experiment']['seed']=int(payload['config']['experiment'].get('seed',0))
        calibration=json.loads(Path(config['experiment']['calibration']).read_text(encoding='utf-8'))
        remote=config.get('communication',{}).get('mode','local')=='http'
        manifest=load_manifest(config['experiment']['manifest'],strict_files=not remote)
        if remote:
            from .worker import RemoteClientExecutor
            identity=hashlib.sha256((str(Path(args.checkpoint).resolve())+'\n'+str(Path(args.output).resolve())).encode()).hexdigest()
            executor=RemoteClientExecutor(config,manifest,identity)
            metrics,_=evaluate_remote_models(model,router,manifest,payload['states'],calibration,config,args.output,executor,
                client_models=payload.get('client_models'),client_bn=payload.get('client_bn'),
                local_new_client_id=payload.get('local_new_client_id'))
        else:
            metrics=evaluate_model(model,router,manifest,payload['states'],calibration,config,Path(args.output)/'predictions',
                                   client_models=payload.get('client_models'),client_bn=payload.get('client_bn'),
                                   local_new_client_id=payload.get('local_new_client_id'))
        write_json(Path(args.output)/'metrics.json',metrics)
        print(json.dumps(metrics,indent=2))
        return 0
    if args.command=='smoke':
        import torch
        from .software_fixture import create_fixture
        try:
            device=torch.device(args.device)
        except (RuntimeError, ValueError) as error:
            parser().error(str(error))
        if device.type not in {'cpu','cuda'}:
            parser().error('smoke supports CPU and CUDA devices')
        if args.amp and device.type!='cuda':
            parser().error('--amp requires a CUDA device')
        if device.type=='cuda':
            if not torch.cuda.is_available():
                parser().error('CUDA is unavailable in this environment')
            if device.index is not None and device.index>=torch.cuda.device_count():
                parser().error('CUDA device index is out of range')
        root=Path(args.output).resolve()
        root.mkdir(parents=True,exist_ok=False)
        torch.set_num_threads(2)
        config=create_fixture(root/'fixture')
        config['experiment']['device']=args.device
        config['optimizer']['amp']=args.amp
        if args.amp:
            # The fixture has few batches for scale calibration.
            config['optimizer']['amp_init_scale']=128.
        result=run_experiment(config,seed=0,output_dir=root/'run')
        write_json(root/'smoke_status.json',{'data_origin':'software_fixture',
                   'rounds':len(result['rounds']),'device':args.device,'amp':args.amp,
                   'output':result['output_dir']})
        print(f'Software smoke test finished: {root}')
        return 0
    raise AssertionError(args.command)


if __name__=='__main__':
    sys.exit(main())
