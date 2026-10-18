"""Evaluation identities and frozen inference protocol."""
from __future__ import annotations

import hashlib
import inspect
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import __version__


def _canonical(value):
    if hasattr(value,'detach'):
        value=value.detach().cpu().numpy()
    if isinstance(value,np.ndarray):
        return {'dtype':str(value.dtype),'shape':list(value.shape),
                'sha256':hashlib.sha256(value.tobytes(order='C')).hexdigest()}
    if isinstance(value,np.generic):
        return value.item()
    if isinstance(value,dict):
        return {str(key):_canonical(item) for key,item in sorted(value.items(),key=lambda item:str(item[0]))}
    if isinstance(value,(tuple,list)):
        return [_canonical(item) for item in value]
    if isinstance(value,Path):
        return str(value)
    return value


def identity_sha256(value):
    payload=json.dumps(_canonical(value),sort_keys=True,separators=(',',':'),allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


@lru_cache(maxsize=8)
def _implementation_sha256(functions):
    return identity_sha256({f'{function.__module__}.{function.__name__}':inspect.getsource(function) for function in functions})


def transform_protocol(config,view):
    from . import data,model,engine
    functions=(data.letterbox_parameters,data.map_boxes,data.load_sample,data.apply_unseen,
               model.sample_tensor,engine.apply_input_policy,engine.prepared_sample)
    evaluation=config.get('evaluation',{})
    perturbation=evaluation.get('unseen',{'rgb_brightness':evaluation.get('unseen_brightness',.65)}) if view=='unseen' else None
    return {'schema_version':1,'implementation_version':__version__,
            'implementation_sha256':_implementation_sha256(functions),'image_size':config.get('model',{}).get('image_size',640),
            'rgb_interpolation':'bilinear','depth_interpolation':'nearest','nir_interpolation':'bilinear',
            'padding':{'RGB':[114,114,114],'depth':0,'NIR':0},'unseen':perturbation,
            'seed':int(config.get('experiment',{}).get('seed',0)),
            'disable_auxiliary':bool(config.get('model',{}).get('disable_auxiliary',False)),
            'detector_rgb_only':bool(config.get('model',{}).get('detector_rgb_only',False)),
            'task':config.get('task')}


def record_metadata(record):
    metadata=record.metadata
    return {'id':str(record.id),'client_id':str(record.client_id),'split':record.split,
            'paired_group_id':metadata.get('paired_group_id'),
            'condition_tag':metadata.get('condition_tag'),
            'condition_tags':metadata.get('condition_tags'),
            'tag_definition_version':metadata.get('tag_definition_version'),
            'condition_tags_version':metadata.get('condition_tags_version')}


def evaluation_record(record,sample,config,view,*,state_source):
    protocol=transform_protocol(config,view)
    metadata=sample.get('meta',{})
    geometry={key:metadata[key] for key in ('original_size','size','resized_size','scale','scale_xy','pad') if key in metadata}
    return {**record_metadata(record),'letterbox':geometry,'applied_unseen':metadata.get('unseen'),
            'transform_sha256':identity_sha256(protocol),'state_source':state_source}


def validate_records(records,manifest,config,view,expected_ids):
    from .data import letterbox_parameters
    if len(records)!=len(expected_ids) or {row['id'] for row in records}!=set(expected_ids):
        raise ValueError('Evaluation audit identities differ')
    protocol=transform_protocol(config,view)
    expected_transform=identity_sha256(protocol)
    by_id=manifest.by_id
    for row in records:
        if any(row.get(key)!=value for key,value in record_metadata(by_id[row['id']]).items()):
            raise ValueError('Evaluation audit manifest metadata differs')
        if row.get('transform_sha256')!=expected_transform:
            raise ValueError('Evaluation transform protocol differs')
        geometry=row.get('letterbox')
        size=config.get('model',{}).get('image_size',640)
        expected_size=[size,size] if isinstance(size,int) else list(size)
        if geometry is None or list(geometry.get('size',[]))!=expected_size:
            raise ValueError('Evaluation letterbox geometry differs')
        try:
            expected_geometry=letterbox_parameters(tuple(geometry['original_size']),size).as_dict()
        except (KeyError,TypeError,ValueError) as exc:
            raise ValueError('Evaluation letterbox geometry differs') from exc
        if identity_sha256(geometry)!=identity_sha256(expected_geometry):
            raise ValueError('Evaluation letterbox geometry differs')
        expected_unseen={'identity':row['id'],'seed':protocol['seed'],'perturbation':protocol['unseen']} if view=='unseen' else None
        if identity_sha256(row.get('applied_unseen'))!=identity_sha256(expected_unseen):
            raise ValueError('Evaluation applied perturbation differs')


def build_evaluation_manifest(model,router,manifest,states,calibration,config,records_by_view,*,
                              client_models=None,client_bn=None,local_new_client_id=None,prediction_dir=None):
    method=config.get('experiment',{}).get('method','tefed')
    prior=calibration.get('state_prior',{'q':.5,'p':.5,'h':[.2]*5})
    views={}
    model_digests={str(client):identity_sha256(values) for client,values in (client_models or {}).items()}
    bn_digests={str(client):identity_sha256(values) for client,values in (client_bn or {}).items()}
    state_digests={str(client):identity_sha256(values) for client,values in states.items()}
    prior_digest=identity_sha256(prior)
    specified={str(key).lower().replace('_','-'):list(map(str,value)) for key,value in manifest.metadata.get('views',{}).items()}
    for view,records in records_by_view.items():
        expected_ids=specified.get(view,[record.id for record in manifest.by_split('hard' if view=='hard' else 'test')])
        validate_records(records,manifest,config,view,expected_ids)
        annotated=[]
        for record in records:
            row=dict(record);client=row['client_id']
            row['state_sha256']=prior_digest if view=='new-client' or client not in states else state_digests[client]
            if client_models:
                selected=local_new_client_id if view=='new-client' else client
                selected=selected if selected in client_models else local_new_client_id
                row['model_selector']={'kind':'client_model','client_id':selected,
                                       'sha256':model_digests[selected]}
            elif client_bn:
                row['model_selector']={'kind':'global_with_client_bn' if view!='new-client' and client in client_bn else 'global',
                    'bn_sha256':bn_digests[client] if view!='new-client' and client in client_bn else None}
            else:
                row['model_selector']={'kind':'global'}
            annotated.append(row)
        views[view]={'records':sorted(annotated,key=lambda row:row['id']),
                     'transform':transform_protocol(config,view),
                     'state_source':'calibration_prior' if view=='new-client' else 'frozen_client_states'}
    prediction_files=None
    if prediction_dir is not None:
        prediction_files={}
        for view in views:
            path=Path(prediction_dir)/f'{view}.jsonl'
            with path.open('rb') as stream:
                digest=hashlib.file_digest(stream,'sha256').hexdigest()
            prediction_files[view]={'path':path.name,'sha256':digest,'bytes':path.stat().st_size}
    return {'schema_version':1,'software_version':__version__,'seed':int(config.get('experiment',{}).get('seed',0)),
            'prediction_files':prediction_files,
            'coordinate_system':'letterbox-normalized-xyxy','head_mode':'general-plus-top2' if method in {'tefed','generic_hc','server_moe'} else 'general',
            'normalization':{'RGB':'float32 / 255','depth':'clip(sensor_units / depth_scale, 0, 1)',
                'depth_scale':float(config.get('model',{}).get('depth_scale',1.)),
                'NIR':'uint8 / 255 after loading','missing_sensor_channels':0},
            'label_dictionary':{'class_names':manifest.metadata.get('class_names'),
                'class_ids':list(range(int(config.get('model',{}).get('classes',2)))),
                'version':manifest.metadata.get('label_dictionary_version'),
                'sha256':identity_sha256(manifest.metadata.get('class_names'))},
            'condition_tag':manifest.metadata.get('condition_tag'),
            'condition_tags':manifest.metadata.get('condition_tags'),
            'tag_definition_version':manifest.metadata.get('tag_definition_version'),
            'condition_tags_version':manifest.metadata.get('condition_tags_version'),
            'config_sha256':identity_sha256(config),'calibration_sha256':identity_sha256(calibration),
            'manifest_sha256':identity_sha256(manifest.as_dict()),
            'frozen_states_sha256':identity_sha256(states),'calibration_prior_sha256':identity_sha256(prior),
            'model_identity':{'kind':'state_dict_sha256','sha256':identity_sha256(model.state_dict()),
                'router_sha256':identity_sha256(router.state_dict()),
                'client_models_sha256':identity_sha256(client_models) if client_models else None,
                'client_bn_sha256':identity_sha256(client_bn) if client_bn else None,
                'local_new_client_id':local_new_client_id},'views':views}
