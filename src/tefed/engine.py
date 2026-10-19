"""Coordinate TE-Fed training, evaluation, and round recovery."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import time
import tomllib

import numpy as np
import torch

from . import IMPLEMENTATION_STATUS
from . import core
from .state import initialize_state, update_state, state_specialist_probabilities
from .data import load_manifest, load_sample, apply_unseen
from .model import RoutedYOLO, Router, sample_tensor, default_prototypes
from .losses import WeightedNativeLoss, soft_consistency, contrastive_alignment
from .rendering import BlenderRenderer
from .diagnostics import renderer_target_observation
from .resume import atomic_snapshot,load_snapshot,locked_run,run_binding,file_journal,reconcile_files


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, default=json_default, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def append_jsonl(path, data):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(data, default=json_default, allow_nan=False) + '\n')


def load_config(path):
    with Path(path).open('rb') as f:
        config = tomllib.load(f)
    config['_config_path'] = str(Path(path).resolve())
    config.setdefault('experiment', {})
    config.setdefault('model', {})
    config.setdefault('decision', {})
    config.setdefault('optimizer', {})
    config.setdefault('loss', {})
    return config


def validate_run_config(config):
    """Validate configuration before loading data or creating outputs."""
    from .baselines import validate_baseline_config
    validate_baseline_config(config)
    if config.get('communication',{}).get('mode','local') not in {'local','http'}:
        raise ValueError('communication.mode must be local or http')
    experiment = config.get('experiment', {})
    for key, default in [('rounds', 30), ('local_epochs', 2), ('batch_size', 4)]:
        value = experiment.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'experiment.{key} must be a positive integer')
    model = config.get('model', {})
    classes = model.get('classes', 2)
    if isinstance(classes, bool) or not isinstance(classes, int) or classes < 2:
        raise ValueError('model.classes must be an integer of at least 2')
    if model.get('head_loss_policy', 'route_weighted') != 'route_weighted':
        raise ValueError('Only the documented route_weighted head loss is implemented')
    size = model.get('image_size', 640)
    if isinstance(size, bool) or not isinstance(size, int) or size < 32 or size % 32:
        raise ValueError('model.image_size must be a positive multiple of 32')
    depth_scale = float(model.get('depth_scale', 1.0))
    if not math.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError('model.depth_scale must be finite and positive')
    cadence=config.get('evaluation',{}).get('monitor_every',0)
    if isinstance(cadence,bool) or not isinstance(cadence,int) or cadence<0:
        raise ValueError('evaluation.monitor_every must be a nonnegative integer')
    scale=config.get('optimizer',{}).get('amp_init_scale',65536.)
    if isinstance(scale,bool) or not isinstance(scale,(int,float)) or not math.isfinite(scale) or scale<=0:
        raise ValueError('optimizer.amp_init_scale must be finite and positive')


def validate_manifest_config(config,manifest):
    if config.get('experiment',{}).get('test_mode',False) and manifest.metadata.get('data_origin')!='generated':
        raise ValueError('experiment.test_mode requires manifest.data_origin=generated')
    classes=config.get('model',{}).get('classes',2)
    names=manifest.metadata.get('class_names')
    if names is not None and len(names)!=classes:
        raise ValueError('model.classes must match manifest.class_names')
    for record in manifest.records:
        if any(label.class_id>=classes for label in record.labels):
            raise ValueError(f'Sample {record.id!r}: class_id must be below model.classes={classes}')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def source_secret(config, *, environ=None):
    """Read the private source-ID key, or use the fixture key in test mode."""
    test_mode = config.get('experiment', {}).get('test_mode', False)
    if not isinstance(test_mode, bool):
        raise ValueError('experiment.test_mode must be a boolean')
    if test_mode:
        return b'tefed-reference-fixture-source-identifiers'
    values = os.environ if environ is None else environ
    value = values.get('TEFED_SOURCE_SECRET')
    if not isinstance(value, str) or len(value.strip().encode('utf-8')) < 32:
        raise ValueError('Actual-data runs require TEFED_SOURCE_SECRET with at least 32 UTF-8 bytes; keep it outside configuration and logs')
    return value.strip().encode('utf-8')


def device_for(config):
    device = str(config.get('experiment', {}).get('device', 'cpu'))
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable; explicitly select cpu for CPU runs')
    return device


def apply_input_policy(sample,config):
    """Apply the configured sensor inputs to every loading path."""
    sample=dict(sample)
    model=config.get('model',{})
    sample['depth_scale']=float(model.get('depth_scale',1.0))
    if model.get('disable_auxiliary',False):
        sample['depth']=None; sample['nir']=None; sample['ir']=None
        sample['enabled_modalities']=['RGB']
    elif config.get('task')=='dronevehicle':
        sample['depth']=None
        sample['ir']=sample.get('nir')
        sample['enabled_modalities']=['RGB','IR']
    return sample


def prepared_sample(record,config):
    sample=load_sample(record,size=config['model'].get('image_size',640))
    return apply_input_policy(sample,config)


def tree_delta(after, before):
    return {'backbone': {k: after['backbone'][k]-v for k,v in before['backbone'].items()},
            'heads': {m:{k:after['heads'][m][k]-v for k,v in head.items()} for m,head in before['heads'].items()}}


def tree_add(base, delta, *, head=None):
    output = deepcopy(base)
    for key, value in delta['backbone'].items():
        output['backbone'][key] = base['backbone'][key] + value
    for m in base['heads']:
        if head is None or str(head) == m:
            for key,value in delta['heads'][m].items():
                output['heads'][m][key] = base['heads'][m][key] + value
    return output


def _strong_view(sample, seed):
    rng = np.random.default_rng(seed)
    result = dict(sample)
    image = np.asarray(sample['rgb']).astype(np.float32)
    result['rgb'] = np.clip(image * rng.uniform(.65, 1.25) + rng.normal(0, 4, image.shape), 0, 255).astype(np.uint8)
    return result


def _modal_inputs(x, sample):
    output = {}
    rgb = x.clone(); rgb[:,3:] = 0
    output['RGB'] = rgb
    if sample.get('depth') is not None:
        depth = torch.zeros_like(x); depth[:,3] = x[:,3]
        output['D'] = depth
    if sample.get('nir') is not None:
        nir = torch.zeros_like(x); nir[:,4] = x[:,4]
        output['NIR'] = nir
    return output


@torch.no_grad()
def assess_frame(model, router, sample, state, calibration, config, *, round_id=0, experts=True, policy=None):
    from .cues import extract_cues, routing_vector
    from .baselines import policy_dict
    if policy is None:
        policy=policy_dict(config.get('experiment',{}).get('method','tefed'),round_id,0,config)
    threshold=policy['pseudo_threshold']
    sample=apply_input_policy(sample,config)
    device = next(model.parameters()).device
    rgb_only = config['model'].get('detector_rgb_only',False)
    x = sample_tensor(sample, device, rgb_only=rgb_only)
    tc = float(calibration.get('T_C', 1))
    confidence = float(config.get('evaluation',{}).get('prediction_confidence',.01))
    maximum = int(config.get('evaluation',{}).get('max_detections',100))
    was_training = model.training
    model.eval()
    try:
        features = model.features(x)
        anchors = model.predict_heads(x, (0,), confidence=confidence, max_detections=maximum,
                                      temperature=tc, cached_features=features)[0][0]
        proposal_count = model.last_proposal_counts[0][0]
        modal_sample = {**sample,'depth':None,'nir':None} if config['model'].get('disable_auxiliary',False) else sample
        modality_tensors=_modal_inputs(sample_tensor(sample,device),modal_sample)
        if rgb_only:
            for name,channel in [('D',3),('NIR',4)]:
                if name in modality_tensors:
                    gray=modality_tensors[name][:,channel:channel+1]
                    branch=torch.zeros_like(x); branch[:,:3]=gray.expand(-1,3,-1,-1)
                    modality_tensors[name]=branch
        modalities = {name:model.predict_heads(value,(0,),confidence=confidence,max_detections=maximum,
                                               temperature=tc)[0][0] for name,value in modality_tensors.items()}
        seed = core.variant_seed(0,round_id,str(sample['id']),0,0)
        strong = _strong_view(sample,seed)
        refs = model.predict_heads(sample_tensor(strong,device,rgb_only=rgb_only), (0,), confidence=confidence,
                                   max_detections=maximum,temperature=tc)[0][0]
        evidence_sample = {**sample,'proposal_count':proposal_count,'proposal_count_includes_target':True}
        if config['model'].get('disable_auxiliary',False):
            evidence_sample['enabled_modalities']=['RGB']
        candidates, vectors, cache = [], [], {}
        for i, anchor in enumerate(anchors):
            pre = extract_cues(evidence_sample, anchor, modalities, [refs], calibration)
            vector = routing_vector(pre,state)
            if experts:
                probabilities = router(torch.tensor([vector], device=device, dtype=torch.float32))[0].tolist()
                route = core.select_route(probabilities)
                outputs = {}
                for m in route['route']:
                    if m == 0:
                        outputs[m] = [anchor]
                    else:
                        if m not in cache:
                            cache[m] = model.predict_heads(x,(m,),confidence=confidence,max_detections=maximum,
                                                           temperature=tc,cached_features=features)[m][0]
                        outputs[m] = cache[m]
                fused = core.fuse_candidate(anchor,outputs,route['route'],route['weights'])
                matched=[(weight,fused['outputs'].get(str(m))) for m,weight in zip(route['route'],route['weights'])]
                matched=[(weight,det) for weight,det in matched if det is not None]
                fused['score']=sum(weight*float(det.get('score',anchor.get('score',0.))) for weight,det in matched)/sum(weight for weight,det in matched)
                post = extract_cues(evidence_sample,fused,modalities,[refs],calibration)
                fused['cues'] = {'C':fused['C'],'AE':fused['AE'],'AM':post['AM'],'G':post['G'],'S':post['S']}
                candidate = core.qualify_candidate(fused,state,config.get('decision'))
            else:
                # General-head baselines use a confidence threshold.
                candidate = {**anchor,'id':str(i),'route':[0],'weights':[1.], 'R':1.,'valid':True,
                             'threshold':threshold,'disposition':'qualified' if max(anchor['posterior'])>=threshold else 'suppressed',
                             'cues':{'C':max(anchor['posterior']),'AE':1.,'AM':pre['AM'],'G':pre['G'],'S':pre['S']}}
                post = pre
            candidate.update(attrs=post['attrs'],availability=post['availability'],rule_flags=post['rule_flags'],
                             routing_vector=vector,pre_cues=pre)
            if config.get('factorial') is not None:
                from .experiments import apply_qualification_control
                candidate=apply_qualification_control(candidate,config)
            candidates.append(candidate)
            vectors.append(vector)
        return {'candidates':candidates,'vectors':vectors,'modalities':modalities,'references':refs}
    finally:
        model.train(was_training)


def validation_loss(model, records, config, head=0, *, samples=None):
    """Evaluate held-out loss without gradients or BN updates."""
    criterion = WeightedNativeLoss(model,head)
    device = next(model.parameters()).device
    training = model.training
    model.eval()
    values=[]
    try:
        with torch.no_grad():
            for record in records:
                sample = prepared_sample(record,config)
                if samples is not None:
                    sample = samples(sample)
                x = sample_tensor(sample,device,rgb_only=config['model'].get('detector_rgb_only',False))
                raw = model(x,(head,))[head]
                value,_ = criterion(raw,sample['targets'],[1.]*len(sample['targets']),device)
                values.append(float(value))
    finally:
        model.train(training)
    if not values:
        raise ValueError('Validation records must be nonempty')
    return float(np.mean(values))


def _state_statistics(samples, assessments, calibration):
    from .cues import quality_stats
    sensors = [quality_stats(sample,calibration) for sample in samples]
    candidates = [c for a in assessments for c in a['candidates']]
    valid = [c for c in candidates if c.get('valid') and c.get('R') is not None]
    failures = np.zeros(5,dtype=int)
    for candidate in candidates:
        if candidate['disposition']!='correction':
            continue
        family_attributes=[3,2,6,1,7]
        triggered=[i for i,flag in enumerate(candidate['rule_flags']) if flag]
        if triggered:
            selected=min(triggered,key=lambda i:(-float(candidate['attrs'][family_attributes[i]] or 0),i))
            failures[selected]+=1
    stats = {}
    for key in ['q_rgb','valid_depth_fraction','q_nir']:
        values = [s[key] for s in sensors if s.get(key) is not None]
        stats[key] = float(np.mean(values)) if values else None
    stats.update(assessed_count=len(candidates),valid_candidate_count=len(valid),
                 qualified_count=sum(c['disposition']=='qualified' for c in candidates),
                 mean_r=float(np.mean([c['R'] for c in valid])) if valid else 0.,
                 mean_s=float(np.mean([c['cues']['S'] for c in valid])) if valid else 0.,
                 mean_ae=float(np.mean([c['cues']['AE'] for c in valid])) if valid else 0.,
                 mean_am_pre=float(np.mean([c['pre_cues']['AM_pre'] for c in candidates])) if candidates else None,
                 failure_counts=failures.tolist())
    return stats


def _crop_embeddings(model,tensors,samples):
    """Collect available regions across the optimizer minibatch."""
    targets=[sample['targets'] for sample in samples]
    if sum(map(len,targets))<2:
        return None,None
    flags=[]
    for sample in samples:
        enabled=sample.get('enabled_modalities')
        flags.append([enabled is None or 'RGB' in enabled,
                      sample.get('depth') is not None and (enabled is None or 'D' in enabled or 'depth' in enabled),
                      sample.get('nir') is not None and (enabled is None or 'NIR' in enabled or 'IR' in enabled)])
    device=tensors[0].device
    available=torch.tensor(flags,device=device,dtype=torch.bool)
    candidates=torch.repeat_interleave(available,torch.tensor([len(t) for t in targets],device=device),dim=0)
    if not any(bool((candidates[:,a]&candidates[:,b]).any()) and int(candidates[:,b].sum())>1
               for a in range(3) for b in range(3) if a!=b):
        return None,None
    return model.contrastive_regions(torch.cat(tensors),targets,available)


def _head_weights(item,head):
    if item['kind']=='labeled':
        weights=item['route_weights']
        return [float(weights.get(head,0))]*len(item['sample']['targets'])
    if item['kind']=='correction':
        d=item['case']['d']
        weights=dict(zip(d['route'],d['weights']))
        return [weights.get(head,0)]*len(item['sample']['targets'])
    return [float(c['R'])*dict(zip(c['route'],c['weights'])).get(head,0.) for c in item['candidates']]


def train_local(model,items,config,*,round_id=0,method='tefed',teacher=None,reference=None):
    from .baselines import policy_dict,update_ema_teacher,fedsto_set_trainable,proximal_penalty,elastic_penalty,orthogonality_penalty
    policies=[policy_dict(method,round_id,epoch,config) for epoch in range(int(config['experiment'].get('local_epochs',2)))]
    device=next(model.parameters()).device
    optimcfg=config.get('optimizer',{})
    total_rounds=int(config['experiment'].get('rounds',30))
    warmup=int(optimcfg.get('warmup_rounds',3))
    initial=float(optimcfg.get('learning_rate',.01)); final=float(optimcfg.get('final_learning_rate',.001))
    if round_id<warmup:
        lr=initial*(round_id+1)/max(warmup,1)
    else:
        progress=(round_id-warmup)/max(1,total_rounds-warmup-1)
        lr=final+(initial-final)*(1+math.cos(math.pi*progress))/2
    optimizer=torch.optim.SGD(model.parameters(),lr=lr,momentum=optimcfg.get('momentum',.937),weight_decay=optimcfg.get('weight_decay',.0005))
    amp=bool(optimcfg.get('amp',True)) and str(device).startswith('cuda')
    scaler=torch.amp.GradScaler('cuda',enabled=amp,init_scale=float(optimcfg.get('amp_init_scale',65536.)))
    counts={'labeled':sum(i['kind']=='labeled' for i in items),
            'pseudo':sum(len(i.get('candidates',[])) for i in items if i['kind']=='pseudo'),
            'correction':sum(i['kind']=='correction' for i in items)}
    losses=config.get('loss',{})
    coefficients={'labeled':1.,'pseudo':float(losses.get('alpha',1.)), 'correction':float(losses.get('beta',.5))}
    criteria={m:WeightedNativeLoss(model,m) for m in range(len(model.heads))}
    for criterion in criteria.values():
        for _ in range(round_id*int(config['experiment'].get('local_epochs',2))):
            criterion.epoch_update()
    batch_size=int(config['experiment'].get('batch_size',4))
    if batch_size<1:
        raise ValueError('batch_size must be positive')
    ledger={'optimizer_steps':0,'attempted_optimizer_steps':0,'amp_skipped_steps':0,'case_presentations':{},
            'presentations':{'labeled':0,'pseudo':0,'correction':0},'mean_step_loss':None,'learning_rate':lr,
            'baseline_policies':[],'teacher_updates':0,'stage_history':[]}
    step_losses=[];alignment_losses=[];consistency_losses=[];matched_consistency=0
    if not items:
        return ledger
    for epoch,policy in enumerate(policies):
        ledger['baseline_policies'].append(deepcopy(policy))
        model.train()
        if method=='fedsto':
            stage=fedsto_set_trainable(model,epoch,head_only_epochs=policy['head_only_epochs'])
            ledger['stage_history'].append({'epoch_index':epoch,**stage})
        order=np.random.permutation(len(items))
        for start in range(0,len(order),batch_size):
            batch=[items[j] for j in order[start:start+batch_size]]
            case_ids=set()
            optimizer.zero_grad(set_to_none=True)
            total=0.
            batch_active=set()
            contrastive_inputs=[];contrastive_samples=[];consistency_sums=[];consistency_count=0
            consistency_coefficient=float(policy['consistency_weight'])
            alignment_enabled=float(losses.get('rho',.1))>0 and not config['model'].get('detector_rgb_only',False) and not config['model'].get('disable_auxiliary',False)
            for item in batch:
                sample=item['sample']; targets=sample['targets']
                active=[m for m in range(len(model.heads)) if any(w>0 for w in _head_weights(item,m))]
                if item['kind']=='labeled' and not targets:
                    active=[m for m,w in item['route_weights'].items() if w>0]
                if not active:
                    raise ValueError('A training item must identify at least one active head')
                batch_active.update(active)
                x=sample_tensor(sample,device,rgb_only=config['model'].get('detector_rgb_only',False))
                with torch.amp.autocast('cuda',enabled=amp):
                    raw=model(x,active)
                    detection=sum(criteria[m](raw[m],targets,_head_weights(item,m),device)[0] for m in active)
                    numerator=len(item.get('candidates',[])) if item['kind']=='pseudo' else 1
                    factor=coefficients[item['kind']]*len(items)*numerator/(counts[item['kind']]+1e-8)
                    loss=detection*factor
                    if alignment_enabled and targets:
                        contrastive_inputs.append(x);contrastive_samples.append(sample)
                    if item['kind']=='pseudo' and consistency_coefficient>0:
                        strong=sample_tensor(_strong_view(sample,round_id*1000+epoch),device,rgb_only=config['model'].get('detector_rgb_only',False))
                        strong_raw=model(strong,(0,))[0]
                        weak_raw=raw.get(0)
                        if teacher is not None:
                            teacher.eval()
                            with torch.no_grad():
                                weak_raw=teacher(x,(0,))[0]
                                weak_raw=weak_raw[1] if isinstance(weak_raw,tuple) else weak_raw
                        if weak_raw is not None:
                            with torch.no_grad():
                                weak_decoded=model.decode_head(weak_raw,0,x.shape[-2:])
                            strong_decoded=model.decode_head(strong_raw,0,strong.shape[-2:])
                            summed,pair_count=soft_consistency(weak_decoded,strong_decoded,
                                confidence=float(config.get('evaluation',{}).get('prediction_confidence',.01)),
                                max_detections=int(config.get('evaluation',{}).get('max_detections',100)),
                                temperature=float(losses.get('consistency_temperature',1.)),return_details=True)
                            if pair_count:
                                consistency_sums.append(summed);consistency_count+=pair_count
                    loss=loss/len(batch)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError('Nonfinite loss; optimizer step aborted')
                scaler.scale(loss).backward()
                total+=float(loss.detach())
                ledger['presentations'][item['kind']]+=numerator
                if item['kind']=='correction':
                    case_id=str(item['case']['id'])
                    tracked=ledger['case_presentations'].setdefault(case_id,{
                        'presentations':0,'supervised_presentations':0,'attempted_step_ids':[],
                        'applied_step_ids':[],'amp_skipped_step_ids':[],
                        'detection_supervision_enabled':coefficients['correction']>0})
                    tracked['presentations']+=numerator
                    tracked['supervised_presentations']+=int(coefficients['correction']>0 and bool(targets))
                    case_ids.add(case_id)
            auxiliary=[]
            with torch.amp.autocast('cuda',enabled=amp):
                if alignment_enabled and contrastive_samples:
                    embeddings,availability=_crop_embeddings(model,contrastive_inputs,contrastive_samples)
                    if embeddings is not None:
                        alignment=contrastive_alignment(embeddings,float(losses.get('temperature',.1)),availability=availability)
                        alignment_losses.append(float(alignment.detach()))
                        auxiliary.append(float(losses.get('rho',.1))*alignment)
                if consistency_count:
                    consistency=sum(consistency_sums)/consistency_count
                    consistency_losses.append(float(consistency.detach()))
                    matched_consistency+=consistency_count
                    auxiliary.append(consistency_coefficient*consistency)
            if auxiliary:
                auxiliary_loss=sum(auxiliary)
                if not torch.isfinite(auxiliary_loss).item():
                    raise FloatingPointError('Nonfinite auxiliary loss; optimizer step aborted')
                scaler.scale(auxiliary_loss).backward()
                total+=float(auxiliary_loss.detach())
            if reference is not None:
                penalties=[]
                if policy.get('proximal_mu',0.)>0:
                    penalties.append(proximal_penalty(model,reference,policy['proximal_mu']))
                if policy.get('elastic_coefficient',0.)>0:
                    penalties.append(elastic_penalty(model,reference,policy['elastic_coefficient']))
                if policy.get('orthogonal_coefficient',0.)>0:
                    penalties.append(orthogonality_penalty(model,coefficient=policy['orthogonal_coefficient'],active_heads=sorted(batch_active)))
                if penalties:
                    penalty=sum(penalties)
                    if not torch.isfinite(penalty).item():
                        raise FloatingPointError('Nonfinite penalty; optimizer step aborted')
                    scaler.scale(penalty).backward()
                    total+=float(penalty.detach())
            scaler.unscale_(optimizer)
            finite_gradients=all(parameter.grad is None or torch.isfinite(parameter.grad).all().item()
                                 for parameter in model.parameters())
            if not finite_gradients and not amp:
                raise FloatingPointError('Nonfinite gradients; optimizer step aborted')
            # GradScaler skips overflowed AMP steps and reduces its scale.
            if finite_gradients:
                torch.nn.utils.clip_grad_norm_(model.parameters(),float(optimcfg.get('gradient_clip',10.)))
            prior_scale=float(scaler.get_scale())
            scaler.step(optimizer); scaler.update()
            ledger['attempted_optimizer_steps']+=1
            executed=not amp or float(scaler.get_scale())>=prior_scale
            step_id=ledger['attempted_optimizer_steps']-1
            for case_id in sorted(case_ids):
                tracked=ledger['case_presentations'][case_id]
                tracked['attempted_step_ids'].append(step_id)
                tracked['applied_step_ids' if executed else 'amp_skipped_step_ids'].append(step_id)
            if teacher is not None and executed:
                update_ema_teacher(teacher,model,float(policy['teacher_decay']))
                ledger['teacher_updates']+=1
            ledger['optimizer_steps']+=int(executed)
            ledger['amp_skipped_steps']+=int(not executed)
            step_losses.append(total)
        for criterion in criteria.values():
            criterion.epoch_update()
    ledger['mean_step_loss']=float(np.mean(step_losses)) if step_losses else None
    ledger.update(mean_contrastive_loss=float(np.mean(alignment_losses)) if alignment_losses else None,
                  mean_consistency_loss=float(np.mean(consistency_losses)) if consistency_losses else None,
                  contrastive_batches=len(alignment_losses),consistency_pairs=matched_consistency)
    return ledger


def save_checkpoint(path,model,router,config,states,pool,round_index):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    atomic_snapshot(path,{'model':model.state_dict(),'router':router.state_dict(),'config':config,
                'states':states,'round_index':round_index,'completed_rounds':round_index+1,
                'model_schema_version':model.schema_version,'implementation_status':IMPLEMENTATION_STATUS})
    write_json(path.with_suffix('.pool.json'),pool)


def load_checkpoint(path,config=None,device='cpu'):
    checkpoint=torch.load(path,map_location=device,weights_only=True)
    if checkpoint.get('model_schema_version')!=RoutedYOLO.schema_version:
        raise ValueError('Checkpoint model schema is incompatible; regenerate it with the current pretrain command')
    cfg=checkpoint['config'] if config is None else config
    construction=deepcopy(cfg)
    construction['model']['weights']=''
    model=RoutedYOLO(construction).to(device)
    model.load_state_dict(checkpoint['model'],strict=True)
    router=Router(cfg.get('decision',{}).get('routing_temperature',.5)).to(device)
    router.load_state_dict(checkpoint['router'],strict=True)
    return model,router,checkpoint


def inherit_initial(model,path):
    """Load shared detector weights for general or expert-bank models."""
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint.get('model_schema_version')!=model.schema_version:
        raise ValueError('Initial checkpoint model schema is incompatible; regenerate it with the current pretrain command')
    current=model.state_dict()
    weights=checkpoint['model']
    compatible={k:v for k,v in weights.items() if k in current and current[k].shape==v.shape}
    required={k for k in current if k.startswith(('trunk.','fusion.','heads.0.','contrastive_projection.'))}
    if not required.issubset(compatible):
        raise ValueError('Initial checkpoint lacks a compatible trunk, general head or contrastive projection; regenerate it with pretrain')
    model.load_state_dict(compatible,strict=False)
    if not any(k.startswith('heads.1.') for k in compatible):
        source=model.heads[0].detect.state_dict()
        for head in model.heads[1:]:
            head.detect.load_state_dict({k:source[k] for k,v in head.detect.state_dict().items()
                                         if k in source and source[k].shape==v.shape},strict=False)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sample_case(case,config):
    from .data import SampleRecord,parse_annotation
    record=SampleRecord(id=case['id'],client_id='server',split='labeled',rgb=Path(case['rgb']),
                        depth=Path(case['depth']) if case.get('depth') else None,
                        nir=Path(case['nir']) if case.get('nir') else None,
                        labels=tuple(parse_annotation(t) for t in case['targets']))
    sample=prepared_sample(record,config)
    conversion=case.get('depth_unit_conversion',case.get('depth_scale'))
    if sample.get('depth') is not None:
        if conversion is None:
            sample['depth']=None
            sample['rendered_depth_units_unresolved']=True
        else:
            conversion=float(conversion)
            if not math.isfinite(conversion) or conversion<=0:
                raise ValueError('Rendered depth unit conversion must be positive and finite')
            sample['depth']=sample['depth']*conversion
    return sample


def _independent_selection(pool,round_index,count):
    eligible=sorted([h for h in pool if int(h['round'])<round_index],
                    key=lambda h:(-h['fidelity'],-h['d'].get('priority',0),str(h['id'])))
    chosen=[]; patterns={}; sources=set()
    for h in eligible:
        d=h['d']; p=d['pattern']; source=d['source']
        if patterns.get(p,0)>=2 or source in sources:
            continue
        chosen.append(h); sources.add(source); patterns[p]=patterns.get(p,0)+1
        if len(chosen)>=count:
            break
    return chosen if count else []


@torch.no_grad()
def evaluate_model(model,router,manifest,states,calibration,config,output_dir,*,client_models=None,client_bn=None,local_new_client_id=None):
    from .metrics import evaluate_detection
    from .evaluation_manifest import evaluation_record,build_evaluation_manifest
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    views=config.get('evaluation',{}).get('views',['participating','unseen','new-client','hard'])
    method=config['experiment'].get('method','tefed')
    experts=method in {'server_moe','generic_hc','tefed'}
    prior=calibration.get('state_prior',{'q':.5,'p':.5,'h':[.2]*5})
    results={};audit_views={}
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    router_before={k:v.detach().clone() for k,v in router.state_dict().items()}
    for view in views:
        if view not in {'participating','unseen','new-client','hard'}:
            raise ValueError(f'Unknown evaluation view: {view}')
        records=manifest.by_split('hard' if view=='hard' else 'test')
        specified=manifest.metadata.get('views',{})
        normalized={str(k).lower().replace('_','-'):v for k,v in specified.items()}
        if view in normalized:
            allowed=set(map(str,normalized[view]))
            records=tuple(r for r in manifest.records if r.id in allowed)
            if len(records)!=len(allowed):
                raise ValueError(f'Evaluation view {view} contains unresolved identities')
        targets={}; predictions={};audit_views[view]=[]
        path=output_dir/f'{view}.jsonl'
        if path.exists():
            raise FileExistsError(path)
        for record in records:
            if client_models:
                client_key=local_new_client_id if view=='new-client' else str(record.client_id)
                client_key=client_key if client_key in client_models else local_new_client_id
                if client_key not in client_models:
                    raise ValueError('Local-only evaluation needs a fixed model for the cold-start view')
                model.load_state_dict(client_models[client_key],strict=True)
            elif client_bn:
                # Restore frozen global BN, then apply the client's trained BN state.
                sd=model.state_dict()
                all_bn={key for values in client_bn.values() for key in values}
                for key in all_bn:
                    sd[key]=before[key]
                if view!='new-client' and str(record.client_id) in client_bn:
                    sd.update(client_bn[str(record.client_id)])
                model.load_state_dict(sd,strict=True)
            sample=prepared_sample(record,config)
            if view=='unseen':
                sample=apply_unseen(sample,identity=record.id,
                                    perturbation=config.get('evaluation',{}).get('unseen',{'rgb_brightness':config.get('evaluation',{}).get('unseen_brightness',.65)}),
                                    seed=int(config['experiment'].get('seed',0)))
            state=prior if view=='new-client' else states.get(str(record.client_id),prior)
            assessed=assess_frame(model,router,sample,state,calibration,config,experts=experts)
            detections=[{'box':c['box'],'class_id':c['class_id'],'score':float(c['score'])} for c in assessed['candidates']]
            targets[record.id]=sample['targets']; predictions[record.id]=detections
            audit_views[view].append(evaluation_record(record,sample,config,view,
                state_source='calibration_prior' if view=='new-client' else
                    ('frozen_client_state' if str(record.client_id) in states else 'calibration_prior_fallback')))
            append_jsonl(path,{'id':record.id,'predictions':detections,'coordinate_system':'letterbox-normalized-xyxy',
                              'view':view,'state_frozen':True})
        results[view]=evaluate_detection(targets,predictions,max_detections=int(config.get('evaluation',{}).get('max_detections',100)))
    if client_models or client_bn:
        model.load_state_dict(before,strict=True)
    for key,value in model.state_dict().items():
        if not torch.equal(value,before[key]):
            raise RuntimeError(f'Evaluation changed model state: {key}')
    for key,value in router.state_dict().items():
        if not torch.equal(value,router_before[key]):
            raise RuntimeError(f'Evaluation changed router state: {key}')
    write_json(output_dir/'evaluation_manifest.json',build_evaluation_manifest(model,router,manifest,states,calibration,config,
        audit_views,client_models=client_models,client_bn=client_bn,local_new_client_id=local_new_client_id,prediction_dir=output_dir))
    return results


def evaluate_current_models(model,router,manifest,states,calibration,config,output_dir,*,local_models=None,bn_cache=None):
    """Evaluate using the matching client model and BN state."""
    method=config['experiment']['method']
    extra={}
    if method=='local':
        records=[r for r in manifest.records if r.id in manifest.calibration_ids]
        if not records:
            raise ValueError('Local-only cold-start selection requires server calibration identities')
        losses={}
        original=deepcopy(model.state_dict())
        try:
            for client,sd in local_models.items():
                model.load_state_dict(sd,strict=True)
                losses[client]=validation_loss(model,records,config)
        finally:
            model.load_state_dict(original,strict=True)
        chosen=min(losses,key=lambda key:(losses[key],key))
        metrics=evaluate_model(model,router,manifest,states,calibration,config,Path(output_dir)/'predictions',
                               client_models=local_models,local_new_client_id=chosen)
        extra={'checkpoint_kind':'local_bundle','client_models':local_models,'local_new_client_id':chosen}
        write_json(Path(output_dir)/'local_model_selection.json',{'protocol':'minimum server calibration loss; fixed before evaluation',
                                                               'losses':losses,'selected_client':chosen})
    elif method=='fedbn':
        metrics=evaluate_model(model,router,manifest,states,calibration,config,Path(output_dir)/'predictions',client_bn=bn_cache)
        extra={'checkpoint_kind':'fedbn_bundle','client_bn':bn_cache}
    else:
        metrics=evaluate_model(model,router,manifest,states,calibration,config,Path(output_dir)/'predictions')
    write_json(Path(output_dir)/'metrics.json',metrics)
    return metrics,extra


def evaluate_remote_models(model,router,manifest,states,calibration,config,output_dir,executor,*,
                           client_runtime=None,client_models=None,client_bn=None,local_new_client_id=None):
    """Pool authenticated worker predictions under the declared view identities."""
    from .metrics import evaluate_detection
    from .evaluation_manifest import build_evaluation_manifest,validate_records
    directory=Path(output_dir)
    clients=sorted({str(record.client_id) for record in manifest.by_split('labeled')})
    runtime=client_runtime or {};local_models=client_models or {};bn_cache=client_bn or {}
    method=config['experiment'].get('method','tefed')
    grouped={view:{'targets':{},'predictions':{},'audit_records':[]} for view in config.get('evaluation',{}).get('views',['participating','unseen','new-client','hard'])}
    def arrays(values):
        return {key:value.detach().cpu().numpy().copy() if isinstance(value,torch.Tensor) else np.asarray(value).copy()
                for key,value in values.items()}
    extra={};new_client_model=None
    if method=='local':
        chosen=local_new_client_id
        if chosen is None:
            records=[manifest.by_id[uid] for uid in manifest.calibration_ids]
            if not records or any(str(record.client_id)!='server' for record in records):
                raise ValueError('Remote Local-only selection requires independent server calibration records')
            original=deepcopy(model.state_dict());losses={}
            try:
                for client,values in local_models.items():
                    model.load_state_dict(values,strict=True)
                    losses[client]=validation_loss(model,records,config)
            finally:
                model.load_state_dict(original,strict=True)
            chosen=min(losses,key=lambda client:(losses[client],client))
            write_json(directory/'local_model_selection.json',{'protocol':'minimum independent server calibration loss',
                'losses':losses,'selected_client':chosen})
        if chosen not in local_models:
            raise ValueError('Remote Local-only evaluation needs its saved cold-start model')
        new_client_model=arrays(local_models[chosen])
        extra={'checkpoint_kind':'local_bundle','client_models':local_models,'local_new_client_id':chosen}
    elif method=='fedbn':
        extra={'checkpoint_kind':'fedbn_bundle','client_bn':bn_cache}
    for client in clients:
        persistent=deepcopy(runtime.get(client,{}))
        if client in local_models:
            persistent['local_model']=arrays(local_models[client])
        if client in bn_cache:
            persistent['bn']=arrays(bn_cache[client])
        if new_client_model is not None:
            persistent['new_client_model']=new_client_model
        result=executor.execute('evaluate',model,router,calibration,client_id=client,state=states[client],persistent_state=persistent)
        if set(result['views'])!=set(grouped):
            raise ValueError('Worker evaluation views differ')
        for view,data in result['views'].items():
            if set(data['targets'])!=set(data['predictions']) or set(data['targets'])&set(grouped[view]['targets']):
                raise ValueError('Worker evaluation identities overlap or differ')
            for name in ('targets','predictions'):
                grouped[view][name].update(data[name])
            validate_records(data.get('audit_records',[]),manifest,config,view,set(data['targets']))
            if any(row['client_id']!=client for row in data['audit_records']):
                raise ValueError('Worker evaluation audit client differs')
            grouped[view]['audit_records'].extend(data['audit_records'])
    metrics={}
    specified={str(key).lower().replace('_','-'):values for key,values in manifest.metadata.get('views',{}).items()}
    for view,data in grouped.items():
        expected=set(specified.get(view,[record.id for record in manifest.by_split('hard' if view=='hard' else 'test')]))
        if set(data['targets'])!=expected:
            raise ValueError(f'Worker evaluation does not cover the declared {view} identities')
        path=directory/'predictions'/f'{view}.jsonl'
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        for uid in sorted(data['predictions']):
            append_jsonl(path,{'id':uid,'predictions':data['predictions'][uid],
                'coordinate_system':'letterbox-normalized-xyxy','view':view,'state_frozen':True})
        metrics[view]=evaluate_detection(data['targets'],data['predictions'],max_detections=int(config.get('evaluation',{}).get('max_detections',100)))
    write_json(directory/'metrics.json',metrics)
    write_json(directory/'predictions'/'evaluation_manifest.json',build_evaluation_manifest(model,router,manifest,states,calibration,config,
        {view:data['audit_records'] for view,data in grouped.items()},client_models=local_models,
        client_bn=bn_cache,local_new_client_id=extra.get('local_new_client_id',local_new_client_id),prediction_dir=directory/'predictions'))
    return metrics,extra


def validate_calibration_binding(config,calibration):
    """Bind measured calibration to the common initial checkpoint."""
    if config.get('experiment',{}).get('test_mode',False) and calibration.get('data_origin')=='generated':
        return
    initial=config.get('model',{}).get('initial_checkpoint')
    if not initial:
        raise ValueError('Measured calibration requires model.initial_checkpoint')
    expected=calibration.get('initial_checkpoint_sha256')
    if not isinstance(expected,str) or len(expected)!=64 or any(c not in '0123456789abcdefABCDEF' for c in expected):
        raise ValueError('Calibration artifact requires initial_checkpoint_sha256; rerun calibrate with the initial checkpoint')
    with Path(initial).expanduser().open('rb') as stream:
        actual=hashlib.file_digest(stream,'sha256').hexdigest()
    if actual!=expected.lower():
        raise ValueError('Calibration initial_checkpoint_sha256 does not match model.initial_checkpoint')


def validate_renderer_setup(config,policy):
    """Check correction-rendering dependencies before local training."""
    renderer=config.get('renderer',{})
    if not policy['use_corrections'] or not renderer.get('enabled',True) or config.get('decision',{}).get('render_budget',1152)<=0:
        return
    backend=BlenderRenderer(renderer.get('blender','blender'),renderer.get('scene'),
                            timeout=float(renderer.get('timeout',600)),samples=int(renderer.get('samples',8)),
                            script_path=renderer.get('script_path'))
    if not os.access(backend.executable,os.X_OK):
        raise PermissionError(f'Blender executable is not executable: {backend.executable}')
    if backend.scene_path is None:
        raise ValueError('renderer.scene is required when correction rendering is enabled')
    if not backend.scene_path.is_file():
        raise FileNotFoundError(f'Blender scene not found: {backend.scene_path}')
    if backend.scene_path.suffix.lower()!='.blend':
        raise ValueError('renderer.scene must refer to a .blend file')
    if renderer.get('scene_class_map') is not None and not backend.generic_script_path.is_file():
        raise FileNotFoundError(f'Generic rendering script not found: {backend.generic_script_path}')


def _validate_aggregation_state(state,policy,clients,next_round,history):
    if policy['aggregation']!='fedawa_paper':
        if state is not None:
            raise ValueError('Unexpected persistent aggregation state')
        return
    if next_round==0:
        if state is not None:
            raise ValueError('Initial FedAWA aggregation state must be empty')
        return
    if not isinstance(state,dict) or set(state)!={'client_ids','logits'} or state['client_ids']!=clients:
        raise ValueError('FedAWA aggregation state client ordering differs')
    logits=state['logits']
    if not isinstance(logits,list) or len(logits)!=len(clients) or any(
            isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) for value in logits):
        raise ValueError('FedAWA aggregation state logits are invalid')
    if history[-1].get('aggregation',{}).get('logits')!=logits:
        raise ValueError('FedAWA aggregation state differs from the committed ledger')


@locked_run
def run_experiment(config,seed=0,output_dir=None,labels_only=False,resume=None):
    from .baselines import policy_dict,policy_schedule,sample_weighted_aggregate,adaptive_aggregation_weights,batchnorm_state_keys
    from .server import construct_cases,update_router,build_cyclic_routes
    from .client import capture_rng_state,restore_rng_state,execute_client_task
    cfg=deepcopy(config)
    validate_run_config(cfg)
    secret=source_secret(cfg)
    cfg['experiment']['seed']=int(seed)
    seed_everything(seed)
    method='fedavg' if labels_only else cfg['experiment'].get('method','tefed')
    policies=policy_schedule(method,cfg)
    policy=policies[0]
    cfg['experiment']['method']=method
    from .experiments import validate_factorial_config
    validate_factorial_config(cfg)
    if not policy['use_experts']:
        cfg['model']['specialists']=0
    elif cfg['model'].get('specialists',5)!=5:
        raise ValueError('Expert methods require five specialists')
    mode=cfg.get('communication',{}).get('mode','local')
    if mode not in {'local','http'}:
        raise ValueError('communication.mode must be local or http')
    remote_mode=mode=='http'
    if labels_only and remote_mode:
        raise ValueError('pretrain requires communication.mode=local')
    manifest=load_manifest(cfg['experiment']['manifest'],strict_files=not remote_mode)
    validate_manifest_config(cfg,manifest)
    clients=sorted({str(r.client_id) for r in manifest.by_split('labeled')})
    if not clients:
        raise ValueError('No participating clients have labeled data')
    if policy['aggregation']=='adaptive_directional' and (len(clients)<2 or len(clients)*policy['aggregation_cap']<1-1e-12):
        raise ValueError('Adaptive aggregation requires at least two clients and n * aggregation_cap >= 1')
    if not labels_only and not manifest.by_split('test'):
        raise ValueError('Test records are required for evaluation')
    replay=None
    if cfg.get('replay'):
        from .replay import ReplayController
        if labels_only or not policy['use_corrections']:
            raise ValueError('Frozen replay requires a correction-capable training method')
        replay=ReplayController(cfg,clients,int(cfg['experiment'].get('rounds',30)),seed)
    if labels_only:
        calibration={'bootstrap_only':True,'T_C':1.,'state_prior':{'q':.5,'p':.5,'h':[.2]*5}}
    else:
        calibration=json.loads(Path(cfg['experiment']['calibration']).read_text(encoding='utf-8'))
        if calibration.get('data_origin')=='generated' and not cfg['experiment'].get('test_mode',False):
            raise ValueError('Generated calibration requires experiment.test_mode=true')
        validate_calibration_binding(cfg,calibration)
    initial=cfg['model'].get('initial_checkpoint')
    if not initial and not labels_only and not cfg['experiment'].get('allow_untrained_initial',False):
        raise ValueError('Provide model.initial_checkpoint; use pretrain to prepare a shared initial detector')
    if not cfg.get('replay'):
        validate_renderer_setup(cfg,policy)
    device=device_for(cfg)
    out=Path(output_dir or Path(cfg['experiment'].get('output','runs/tefed'))/f'seed_{seed}').resolve()
    executor=None
    if remote_mode:
        from .worker import RemoteClientExecutor
        executor=RemoteClientExecutor(cfg,manifest,hashlib.sha256(str(out).encode()).hexdigest())
    binding=run_binding(cfg,manifest,labels_only=labels_only,secret=secret,
                        workers=executor.fingerprints if executor is not None else None)
    restored=None
    if resume is not None:
        if Path(resume).resolve()!=out/'resume.pt':
            raise ValueError('Resume must use this run directory\'s latest resume.pt')
        restored=load_snapshot(resume)
        if restored.get('resume_schema_version')!=1 or restored.get('output_dir')!=str(out):
            raise ValueError('Unsupported resume snapshot or different run directory')
        if restored.get('binding')!=binding:
            raise ValueError('Resume protocol, input artifacts, source key or runtime changed')
        next_round=restored.get('next_round')
        total_rounds=int(cfg['experiment'].get('rounds',30))
        if type(next_round) is not int or not 0<=next_round<=total_rounds or restored.get('phase') not in {'training','completed'}:
            raise ValueError('Resume round boundary is invalid')
        if restored['phase']=='completed' and next_round!=total_rounds:
            raise ValueError('Completed resume snapshot has an incomplete round sequence')
        history=restored.get('history',[])
        if len(history)!=next_round or [row['round_index'] for row in history]!=list(range(next_round)):
            raise ValueError('Resume history is not contiguous')
        _validate_aggregation_state(restored.get('aggregation_state'),policy,clients,next_round,history)
        reconcile_files(out,restored['files'],restored['next_round'],restored['phase']=='completed')
        if restored['phase']=='completed':
            restore_rng_state(restored['rng'])
            return {'output_dir':str(out),'rounds':restored['history'],'checkpoint':str(out/'final.pt'),'metrics':restored['metrics']}
    else:
        out.mkdir(parents=True,exist_ok=False)
        write_json(out/'status.json',{'status':'started','implementation_status':IMPLEMENTATION_STATUS,'seed':seed,'completed_rounds':0})
    model=RoutedYOLO(cfg).to(device)
    router=Router(cfg.get('decision',{}).get('routing_temperature',.5)).to(device)
    initial_hash=inherit_initial(model,initial) if initial and restored is None else None
    if restored is None:
        write_json(out/'configuration.json',{**cfg,'resolved_baseline_policies':policies})
        write_json(out/'calibration.json',calibration)
        write_json(out/'environment.json',{'implementation_status':IMPLEMENTATION_STATUS,'seed':seed,
                'versions':{k:importlib.metadata.version(k) for k in ['torch','torchvision','ultralytics','numpy']},
                'device':str(device),'initial_checkpoint_sha256':initial_hash,
                'calibration_sha256':hashlib.sha256(Path(cfg['experiment']['calibration']).read_bytes()).hexdigest() if not labels_only else None,
                'manifest_sha256':hashlib.sha256(Path(cfg['experiment']['manifest']).read_bytes()).hexdigest(),
                'capacity':model.capacity(),'router_parameters':sum(parameter.numel() for parameter in router.parameters()),
                'logical_clients':clients,'labels_only_preparation':labels_only,
                'workers':executor.fingerprints if executor is not None else None})
    prior=calibration.get('state_prior',{'q':.5,'p':.5,'h':[.2]*5})
    states={k:deepcopy(prior) for k in clients}
    source_states={}; pool=[]; history=[]; previous_state=None;aggregation_state=None
    if replay is not None:
        source_states=replay.source_states;pool=replay.pool
    local_models={};bn_cache={};client_runtime={}
    start_round=0
    journal={}
    if restored is not None:
        if restored['model_schema_version']!=model.schema_version:
            raise ValueError('Resume model schema changed')
        model.load_state_dict(restored['model'],strict=True)
        router.load_state_dict(restored['router'],strict=True)
        states=restored['states'];source_states=restored['source_states'];pool=restored['pool']
        history=restored['history'];previous_state=restored['previous_state']
        aggregation_state=restored.get('aggregation_state')
        local_models=restored['local_models'];bn_cache=restored['bn_cache'];client_runtime=restored.get('client_runtime',{})
        start_round=restored['next_round'];journal=restored['files']
        if len(history)!=start_round or [row['round_index'] for row in history]!=list(range(start_round)):
            raise ValueError('Resume history is not contiguous')
        restore_rng_state(restored['rng'])
        if replay is not None:
            replay.load_state_dict(restored['replay'])
            replay.verify_history(history)
    def client_task(action,client,**kwargs):
        if executor is not None:
            return executor.execute(action,model,router,calibration,client_id=client,**kwargs)
        return execute_client_task(cfg,manifest,model,router,calibration,action=action,client_id=client,run_secret=secret,**kwargs)
    def evaluate_run(directory):
        if executor is None:
            return evaluate_current_models(model,router,manifest,states,calibration,cfg,directory,
                                           local_models=local_models,bn_cache=bn_cache)
        return evaluate_remote_models(model,router,manifest,states,calibration,cfg,directory,executor,
                                      client_runtime=client_runtime,client_models=local_models,client_bn=bn_cache)
    if restored is None and not labels_only:
        # Initialize state from validation data with frozen initial weights.
        for client in clients:
            states[client]=client_task('initialize',client,state=prior)['state']
    def commit(next_round,*,phase='training',metrics=None):
        nonlocal journal
        if replay is not None:
            replay.verify();replay.verify_history(history)
        journal=file_journal(out,journal,next_round-1 if next_round else None,completed=phase=='completed')
        atomic_snapshot(out/'resume.pt',{'resume_schema_version':1,'model_schema_version':model.schema_version,
            'output_dir':str(out),'binding':binding,'phase':phase,'next_round':next_round,
            'model':model.state_dict(),'router':router.state_dict(),'states':states,'source_states':source_states,
            'pool':pool,'history':history,'previous_state':previous_state,'aggregation_state':aggregation_state,
            'local_models':local_models,
            'bn_cache':bn_cache,'client_runtime':client_runtime,
            'rng':capture_rng_state(),'files':journal,'metrics':metrics or {},
            'replay':replay.state_dict() if replay is not None else None})
        write_json(out/'status.json',{'status':'completed' if phase=='completed' else 'running',
            'implementation_status':IMPLEMENTATION_STATUS,'completed_rounds':next_round,
            'checkpoint':str(out/'final.pt') if phase=='completed' else str(out/'resume.pt')})
    if restored is None:
        commit(0)
    for t in range(start_round,int(cfg['experiment'].get('rounds',30))):
        policy=policy_dict(method,t,0,cfg)
        start_time=time.perf_counter()
        start_ts=time.time()
        if device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(device)
        base=model.export_state()
        states_start=deepcopy(states)
        uploads=[]; trained_states=[]; sample_counts=[]; descriptors=[];round_candidates=[]
        ledger={'round_index':t,'clients':{},'reuse_selected_ids':{},'selected_creation_rounds':{},
                'baseline_policy':deepcopy(policy)}
        if replay is not None:
            ledger['replay']=replay.metadata()
        next_states={}
        for client in clients:
            state=states_start[client]
            replay_control=None
            if replay is not None:
                selection=replay.select(state,client,t,cfg.get('decision'))
                selected=selection['selected'];replay_control=selection['control']
            elif policy['use_corrections'] and not labels_only:
                if cfg.get('decision',{}).get('selection_mode','matched')=='independent':
                    cap=int(cfg.get('decision',{}).get('reuse_cap',6))
                    selected=_independent_selection(pool,t,cap)
                else:
                    selected=core.select_reuse(pool,state,t,cfg.get('decision'))['selected']
            else:
                selected=[]
            if any(h['round']>=t for h in selected):
                raise AssertionError('Same-round correction reuse is forbidden')
            ledger['reuse_selected_ids'][client]=[h['id'] for h in selected]
            ledger['selected_creation_rounds'][client]=[h['round'] for h in selected]
            result=client_task('client_round',client,round_index=t,state=state,selected_cases=selected,
                               persistent_state=client_runtime.get(client),replay_control=replay_control)
            client_runtime[client]=result['persistent_state']
            for name,cache in (('local_model',local_models),('bn',bn_cache)):
                value=result['persistent_state'].get(name)
                if value is not None:
                    cache[client]={key:torch.as_tensor(array.copy()) for key,array in value.items()}
            if replay is None:
                descriptors.extend(result['descriptors'])
                source_states.update(result['source_states'])
            for event in result['selected_events']:
                append_jsonl(out/'selected_events.jsonl',event)
            audit_name='client_'+hashlib.sha256(client.encode('utf-8')).hexdigest()[:16]+'_candidate_audit.jsonl'
            for event in result['candidate_audit']:
                append_jsonl(out/audit_name,event)
            round_candidates.extend(result['candidate_audit'])
            if result['upload'] is not None:
                uploads.append(result['upload'])
            trained_states.append(result['after']);sample_counts.append(result['sample_count'])
            next_states[client]=result['next_state']
            ledger['clients'][client]=result['ledger']
        # Verify against the round-start bank before aggregation.
        if replay is None and descriptors and cfg.get('renderer',{}).get('enabled',True) and cfg.get('decision',{}).get('render_budget',1152)>0:
            cyclic_routes=build_cyclic_routes(round_candidates,manifest,secret,t,run_seed=seed)
            created=construct_cases(descriptors,model,router,source_states,clients,calibration,cfg,t,seed,out/f'rendering/round_{t:03d}',
                                    cyclic_routes=cyclic_routes)
        else:
            created={'cases':[],'audit':[]}
        pool.extend(created['cases'])
        write_json(out/f'verification_round_{t:03d}.json',created['audit'])
        if method=='local':
            # Retain each client's checkpoint for local-only evaluation.
            pass
        elif policy['use_experts']:
            aggregate=core.aggregate_updates(base,uploads,states_start,default_prototypes(),previous_state=previous_state,config=cfg.get('decision'))
            previous_state=base
            model.import_state(aggregate['state'])
            ledger['aggregation']={k:v for k,v in aggregate.items() if k!='state'}
            ledger['aggregation']['baseline_policy']=deepcopy(policy)
        else:
            weights=sample_counts
            aggregation_details={}
            if policy['aggregation']=='adaptive_directional':
                parameter_names=set(dict(model.named_parameters()))
                vectors=[np.concatenate([u['backbone'][k].ravel() for k in sorted(u['backbone']) if k in parameter_names]) for u in [tree_delta(s,base) for s in trained_states]]
                weights=adaptive_aggregation_weights(vectors,
                    sample_counts=sample_counts if policy['reference_weighting']=='sample_count' else None,
                    temperature=policy['aggregation_temperature'],cap=policy['aggregation_cap'])
            elif policy['aggregation']=='fedawa_paper':
                from .fedawa import flatten_parameters,paper_aggregation_weights
                parameter_names=set(dict(model.named_parameters()))
                aggregation_details=paper_aggregation_weights(
                    [flatten_parameters(state,parameter_names) for state in trained_states],
                    flatten_parameters(base,parameter_names),
                    logits=None if aggregation_state is None else aggregation_state['logits'],
                    sample_counts=sample_counts,steps=policy['fedawa_steps'],
                    learning_rate=policy['fedawa_learning_rate'])
                weights=aggregation_details['weights']
                aggregation_state={'client_ids':list(clients),'logits':list(aggregation_details['logits'])}
            keep=batchnorm_state_keys(model) if method=='fedbn' else set()
            aggregate=sample_weighted_aggregate(trained_states,weights,previous=base,keep_predicate=lambda key:key in keep)
            model.import_state(aggregate)
            normalized=np.asarray(weights,dtype=np.float64);normalized=normalized/normalized.sum()
            ledger['aggregation']={**aggregation_details,'baseline_policy':deepcopy(policy),
                                   'weights':dict(zip(clients,normalized.tolist()))}
        if policy['use_experts'] and not labels_only and cfg.get('router',{}).get('steps',20)>0:
            ledger['router']=update_router(model,router,manifest,pool,source_states,states_start,calibration,cfg,t,
                                           descriptors=created.get('scored_descriptors',descriptors))
        else:
            ledger['router']={'updated':False,'reason':'disabled_or_general_only'}
        states=next_states
        ledger.update(new_cases=len(created['cases']),retained_pool_size=len(pool),seconds=time.perf_counter()-start_time,
                      start_ts=start_ts,end_ts=time.time(),
                      server_gpu_peak_bytes=int(torch.cuda.max_memory_allocated(device)) if device.startswith('cuda') else None,
                      energy_joules=None)
        cadence=int(cfg.get('evaluation',{}).get('monitor_every',0))
        if not labels_only and cadence>0 and (t+1)%cadence==0:
            monitor_start=time.perf_counter()
            monitoring,_=evaluate_run(out/f'monitoring/round_{t:03d}')
            append_jsonl(out/'monitoring.jsonl',{'round_index':t,'metrics':monitoring})
            ledger['monitoring_seconds']=time.perf_counter()-monitor_start
        append_jsonl(out/'rounds.jsonl',ledger)
        history.append(ledger)
        commit(t+1)
        print(f'{method} seed={seed} round={t+1}: steps={sum(x["optimizer_steps"] for x in ledger["clients"].values())}, new_cases={len(created["cases"])}',flush=True)
    final=out/'final.pt'
    save_checkpoint(final,model,router,cfg,states,pool,len(history)-1)
    write_json(out/'source_states.json',source_states)
    metrics={}
    if not labels_only:
        metrics,extra=evaluate_run(out)
        if extra:
            checkpoint=torch.load(final,map_location='cpu',weights_only=True)
            checkpoint.update(extra)
            atomic_snapshot(final,checkpoint)
    commit(len(history),phase='completed',metrics=metrics)
    return {'output_dir':str(out),'rounds':history,'checkpoint':str(final),'metrics':metrics}


def pretrain(config,seed=0,output_dir=None,resume=None):
    """Prepare a shared detector with supervised FedAvg on local labels."""
    cfg=deepcopy(config)
    cfg['experiment']['method']='fedavg'
    cfg['model']['specialists']=0
    return run_experiment(cfg,seed=seed,output_dir=output_dir,labels_only=True,resume=resume)
