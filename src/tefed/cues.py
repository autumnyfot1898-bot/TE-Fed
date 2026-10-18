"""Measure cues from observations and predictions without target labels.

Use the general anchor before routing and the fused box afterward.
Missing measurements retain availability=0."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite, log, sqrt
from numbers import Real

import numpy as np

from .state import validate_state

TOMATO_PRIORS={
    "0":{"a":-1.64,"b":-.92,"aspect":1.03},
    "1":{"a":-1.71,"b":-.89,"aspect":.97},
}
FAMILY_ATTRIBUTES=(3,2,6,1,7)
DEFAULT_CONDITION_THRESHOLDS=(.18,.15,.22,.25,.20)


def _clip(value):
    return float(np.clip(value,0.,1.))


def _finite_real(value):
    return not isinstance(value,bool) and isinstance(value,Real) and isfinite(value)


def _positive(value):
    return _finite_real(value) and value>0


def _detection(value):
    try:
        b=np.asarray(value["box"],dtype=float)
        p=np.asarray(value["posterior"],dtype=float)
    except (ValueError,TypeError,KeyError):
        return None
    if (b.shape!=(4,) or not np.isfinite(b).all() or b[2]<=b[0] or b[3]<=b[1]
            or p.ndim!=1 or p.size<2 or not np.isfinite(p).all()
            or np.any(p<0) or np.any(p>1) or not np.isclose(p.sum(),1.,atol=1e-6,rtol=0)):
        return None
    cls=int(np.argmax(p))
    if "class_id" in value and value["class_id"]!=cls:
        return None
    return {"box":b,"posterior":p,"class_id":cls,"index":int(value.get("index",0))}


def _iou(a,b):
    intersect=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
    aa=max(0.,a[2]-a[0])*max(0.,a[3]-a[1])
    bb=max(0.,b[2]-b[0])*max(0.,b[3]-b[1])
    return intersect/(aa+bb-intersect) if aa+bb-intersect>0 else 0.


def _items(outputs):
    if outputs is None:
        return []
    if isinstance(outputs,Mapping):
        if "box" in outputs:
            return [outputs]
        return outputs.get("detections",[])
    return list(outputs)


def _best(outputs,anchor,*,class_required=False,min_iou=0.):
    valid=[]
    for raw in _items(outputs):
        d=_detection(raw)
        if d is None or (class_required and d["class_id"]!=anchor["class_id"]):
            continue
        overlap=_iou(d["box"],anchor["box"])
        if overlap>=min_iou:
            valid.append((overlap,float(d["posterior"].max()),d["index"],d))
    return min(valid,key=lambda x:(-x[0],-x[1],x[2]))[3] if valid else None


def _modality_name(name):
    return {"rgb":"RGB","d":"D","depth":"D","nir":"NIR","ir":"IR"}.get(str(name).lower(),str(name).upper())


def _modalities(outputs):
    result={}
    for name,predictions in outputs.items():
        normalized=_modality_name(name)
        if normalized in result:
            raise ValueError(f"duplicate modality aliases: {name}")
        result[normalized]=predictions
    return result


def _reference_views(reference_outputs):
    if reference_outputs is None:
        return []
    if isinstance(reference_outputs,Mapping):
        if "previous" in reference_outputs and reference_outputs["previous"] is not None:
            return [reference_outputs["previous"]]
        return [reference_outputs[k] for k in ("weak","strong") if reference_outputs.get(k) is not None]
    views=list(reference_outputs)
    previous=[v for v in views if isinstance(v,Mapping) and v.get("kind")=="previous" and v.get("detections") is not None]
    if previous:
        return [previous[0]["detections"]]
    return [v["detections"] if isinstance(v,Mapping) and "detections" in v else _items(v) for v in views if v is not None]


def _crop_indices(box,height,width):
    x1=max(0,min(width,int(np.floor(box[0]*width))))
    x2=max(0,min(width,int(np.ceil(box[2]*width))))
    y1=max(0,min(height,int(np.floor(box[1]*height))))
    y2=max(0,min(height,int(np.ceil(box[3]*height))))
    return y1,y2,x1,x2


def _gray_rgb(rgb):
    x=np.asarray(rgb)
    if x.ndim!=3 or x.shape[2]<3 or x.shape[0]<1 or x.shape[1]<1:
        raise ValueError("rgb must have nonempty HWC RGB channels")
    x=x[...,:3].astype(np.float64)
    if not np.isfinite(x).all() or x.min()<0 or x.max()>255:
        raise ValueError("RGB intensities must be finite and in [0,255]")
    if np.asarray(rgb).dtype.kind=="f" and x.max()<=1:
        x=x*255.
    return x, np.tensordot(x,np.array([.299,.587,.114]),axes=([-1],[0]))


def _median3(image):
    padded=np.pad(image,1,mode="reflect" if min(image.shape)>1 else "edge")
    windows=np.lib.stride_tricks.sliding_window_view(padded,(3,3))
    return np.median(windows,axis=(-2,-1))


def _laplacian_variance(gray):
    p=np.pad(gray,1,mode="reflect" if min(gray.shape)>1 else "edge")
    lap=p[:-2,1:-1]+p[2:,1:-1]+p[1:-1,:-2]+p[1:-1,2:]-4*p[1:-1,1:-1]
    return float(np.var(lap))


def _rgb_degradation(rgb,gray):
    if not gray.size:
        return {"q_rgb":None,"laplacian_variance":None,"clipping_fraction":None}
    variance=_laplacian_variance(gray)
    blur=1. if min(gray.shape)<16 else _clip(1-variance/120)
    clipped=float(np.any((rgb<=0)|(rgb>=255),axis=2).mean())
    return {"q_rgb":_clip(.70*blur+.30*_clip(clipped/.20)),
            "laplacian_variance":variance,"clipping_fraction":clipped}


def _enabled_modalities(sample):
    enabled=sample.get("enabled_modalities")
    if enabled is None:
        enabled=[name for key,name in (("rgb","RGB"),("depth","D"),("nir","NIR"),("ir","IR")) if key in sample]
    result={_modality_name(name) for name in enabled}
    result.add("RGB")
    return result


def _infrared_input(sample,enabled):
    if "NIR" in enabled:
        return sample.get("nir")
    # DroneVehicle stores IR in nir; preserve its modality tag.
    value=sample.get("ir")
    return value if value is not None else sample.get("nir")


def quality_stats(sample:dict,calibration:dict|None=None)->dict:
    """Measure full-frame sensor quality, including frames without detections.

    Disabled sensors return None; failed enabled sensors use depth fraction 0
    or NIR noise 1. mean_am_pre stays None for the state adapter."""
    rgb,gray=_gray_rgb(sample["rgb"])
    enabled=_enabled_modalities(sample)
    measured=_rgb_degradation(rgb,gray)
    depth_fraction=None
    if "D" in enabled:
        raw=sample.get("depth")
        depth=np.asarray(raw) if raw is not None else None
        if depth is not None and depth.ndim==3 and depth.shape[2]==1:
            depth=depth[...,0]
        depth_fraction=float((np.isfinite(depth)&(depth>0)).mean()) if depth is not None and depth.shape==gray.shape else 0.
    nir_noise=None
    if "NIR" in enabled or "IR" in enabled:
        nir_noise,_=_nir_noise(_infrared_input(sample,enabled),gray.shape,(slice(0,gray.shape[0]),slice(0,gray.shape[1])))
    return {"q_rgb":measured["q_rgb"],"valid_depth_fraction":depth_fraction,
            "q_nir":nir_noise,"mean_am_pre":None,
            "enabled_modalities":sorted(enabled),
            "quality_weights":{"q_rgb":.30,"valid_depth_fraction":.30,"q_nir":.20,"mean_am_pre":.20}}


def _nir_noise(nir,crop_shape,slices):
    if nir is None:
        return 1.,False
    image=np.asarray(nir)
    if image.ndim==3:
        image=image[...,0]
    if image.ndim!=2 or image.shape!=crop_shape:
        return 1.,False
    x=image[slices].astype(np.float64)
    if x.size==0 or not np.isfinite(x).all() or x.min()<0 or x.max()>255:
        return 1.,False
    if image.dtype.kind in "ui" or x.max()>1:
        x=x/255.
    residual=x-_median3(x)
    mad=float(np.median(np.abs(residual-np.median(residual))))
    return _clip(1.4826*mad/.08),True


def _depth_measurements(depth,shape,slices,prior,calibration):
    result={"valid_fraction":0.,"depth_residual":None,"perspective":.5,
            "depth_reliable":False,"depth_present":False,"valid_pixels":0}
    if depth is None:
        return result
    image=np.asarray(depth)
    if image.ndim==3 and image.shape[2]==1:
        image=image[...,0]
    if image.ndim!=2 or image.shape!=shape:
        return result
    crop=image[slices].astype(float)
    if not crop.size:
        return result
    valid=np.isfinite(crop)&(crop>0)
    fraction=float(valid.mean())
    result.update(valid_fraction=fraction,valid_pixels=int(valid.sum()),depth_present=True)
    depth_median=calibration.get("depth_validation_median",calibration.get("median_validation_depth"))
    prior_a=prior.get("a",prior.get("a_c")) if prior else None
    prior_b=prior.get("b",prior.get("b_c")) if prior else None
    if fraction>=.35 and valid.any() and _positive(depth_median) and _finite_real(prior_a) and _finite_real(prior_b):
        z=float(np.median(crop[valid]))/float(depth_median)
        predicted_height=exp_safe(float(prior_a)+float(prior_b)*log(max(z,1e-3)))
        # Normalized height is assigned below.
        result.update(depth_reliable=True,predicted_height=predicted_height)
    if fraction>=.35 and valid.sum()>=64:
        values=crop[valid]
        lo,hi=np.percentile(values,[10,90])
        keep=valid&(crop>=lo)&(crop<=hi)
        ys,xs=np.nonzero(keep)
        if len(xs)>=3:
            global_y=ys+slices[0].start
            global_x=xs+slices[1].start
            xx=2*global_x/max(shape[1]-1,1)-1
            yy=2*global_y/max(shape[0]-1,1)-1
            design=np.stack([xx,yy,np.ones_like(xx)],axis=1)
            coefficients,_,rank,_=np.linalg.lstsq(design,crop[keep],rcond=None)
            if rank==3 and np.isfinite(coefficients).all():
                a,b,c=coefficients
                result["perspective"]=_clip(sqrt(a*a+b*b)/(.35*abs(c)+1e-6))
    return result


def exp_safe(value):
    # Reject invalid log-height priors instead of clipping calibration.
    if not isfinite(value) or value>700 or value<-700:
        raise ValueError("invalid calibrated depth-size prior")
    return float(np.exp(value))


def _context_entropy(gray,box):
    height,width=gray.shape
    cx,cy=(box[0]+box[2])/2,(box[1]+box[3])/2
    bw,bh=box[2]-box[0],box[3]-box[1]
    original=np.zeros((height,width),dtype=bool)
    y1,y2,x1,x2=_crop_indices(box,height,width)
    original[y1:y2,x1:x2]=True
    for scale in (1.5,1.25):
        expanded=[cx-scale*bw/2,cy-scale*bh/2,cx+scale*bw/2,cy+scale*bh/2]
        yy1,yy2,xx1,xx2=_crop_indices(expanded,height,width)
        mask=np.zeros((height,width),dtype=bool)
        mask[yy1:yy2,xx1:xx2]=True
        mask &= ~original
        if mask.sum()>=256:
            hist,_=np.histogram(gray[mask],bins=32,range=(0,256))
            probabilities=hist/hist.sum()
            entropy=-float(np.sum(probabilities*np.log(probabilities+1e-12)))/log(32)
            return _clip(entropy),int(mask.sum()),scale
    return None,int(mask.sum()),None


def extract_cues(sample: dict,anchor: dict,modality_outputs: dict,
                 reference_outputs: list | None,calibration: dict) -> dict:
    """Measure pre-route or fused cues from observations and predictions.

    proposal_count counts detections above .05 before NMS, excluding the anchor
    unless proposal_count_includes_target=True. foreground_detections supplies
    occlusion comparators; neither may come from target labels. Flat reference
    lists contain one detection per view; nested lists preserve multiple boxes.
    A previous/weak/strong dict specifies temporal preference."""
    anchor=_detection(anchor)
    if anchor is None:
        raise ValueError("anchor needs a positive box and calibrated class posterior")
    calibration=calibration or {}
    rgb,gray=_gray_rgb(sample["rgb"])
    height,width=gray.shape
    box=anchor["box"]
    y1,y2,x1,x2=_crop_indices(box,height,width)
    slices=(slice(y1,y2),slice(x1,x2))
    crop=gray[slices]
    rgb_crop=rgb[slices]
    outputs=_modalities(modality_outputs)
    enabled=_enabled_modalities(sample)
    if "IR" in enabled and "NIR" not in enabled and "NIR" in outputs and "IR" not in outputs:
        outputs["IR"]=outputs.pop("NIR")
    audit=[]

    # Class conflicts retain their branch weight.
    nominal=calibration.get("modality_weights",{"RGB":.40,"D":.35,"NIR":.25,"IR":.25})
    nominal={_modality_name(k):float(v) for k,v in nominal.items()}
    numerator=denominator=0.
    reference_boxes=[]
    for name,predictions in outputs.items():
        if name not in enabled or predictions is None:
            continue
        branch=_best(predictions,anchor)
        if branch is None:
            # No aligned prediction: mark the modality unavailable.
            continue
        weight=nominal.get(name)
        if weight is None or not isfinite(weight) or weight<0:
            raise ValueError(f"explicit nonnegative modality weight required: {name}")
        denominator+=weight
        if branch["class_id"]==anchor["class_id"]:
            numerator+=weight*_iou(branch["box"],box)
        consistent=_best(predictions,anchor,class_required=True)
        if consistent is not None:
            reference_boxes.append(consistent["box"])
    agreement=numerator/denominator if denominator else 0.

    views=_reference_views(reference_outputs)
    temporal_ious=[]
    for view in views:
        matched=_best(view,anchor,class_required=True,min_iou=.50)
        if matched is not None:
            temporal_ious.append(_iou(matched["box"],box))
        consistent=_best(view,anchor,class_required=True)
        if consistent is not None:
            reference_boxes.append(consistent["box"])
    consistency=float(np.mean(temporal_ious)) if temporal_ious else 0.
    temporal_iou=max(temporal_ious) if temporal_ious else 0.
    dispersion=_clip(sum(float(np.abs(b-box).sum()) for b in reference_boxes)/(4*len(reference_boxes))) if reference_boxes else 0.

    priors=calibration.get("class_priors")
    if priors is None and calibration.get("task","tomato")=="tomato":
        priors=TOMATO_PRIORS
    prior=priors.get(str(anchor["class_id"]),priors.get(anchor["class_id"])) if isinstance(priors,Mapping) else None
    aspect=None
    calibrated_aspect=prior.get("aspect",prior.get("r_c_med")) if prior else None
    if _positive(calibrated_aspect):
        ratio=(box[2]-box[0])*width/((box[3]-box[1])*height)
        aspect=_clip(abs(log((ratio+1e-6)/(float(calibrated_aspect)+1e-6)))/log(1.7))
    else:
        audit.append("aspect_calibration_missing")
    depth=_depth_measurements(sample.get("depth") if "D" in enabled else None,(height,width),slices,prior,calibration)
    if depth["depth_reliable"]:
        depth["depth_residual"]=_clip(abs(log(((box[3]-box[1])+1e-6)/(depth["predicted_height"]+1e-6)))/log(1.8))
    elif "D" in enabled:
        # SI uses residual=1 for insufficient depth; uncalibrated depth uses geometry.
        depth["depth_residual"]=1. if depth["valid_fraction"]<.35 else None
        if not _positive(calibration.get("depth_validation_median",calibration.get("median_validation_depth"))):
            audit.append("depth_calibration_missing")

    bw,bh=box[2]-box[0],box[3]-box[1]
    clipped_width=max(0.,min(1.,box[2])-max(0.,box[0]))
    clipped_height=max(0.,min(1.,box[3])-max(0.,box[1]))
    area=_clip(clipped_width*clipped_height/(bw*bh))
    geometry=None
    if aspect is not None:
        if depth["depth_reliable"]:
            geometry=_clip(.30*area+.25*(1-aspect)+.45*(1-depth["depth_residual"]))
        else:
            geometry=_clip(.55*area+.45*(1-aspect))
    rgb_geometry=("D" not in enabled)
    view=aspect if rgb_geometry else (_clip(.60*depth["perspective"]+.40*aspect) if aspect is not None else None)

    attrs=[None]*8
    attrs[0]=_clip(1-sqrt(bw*bh)/.35)
    sorted_p=np.sort(anchor["posterior"])
    attrs[1]=_clip(1-(sorted_p[-1]-sorted_p[-2]))

    foreground=sample.get("foreground_detections",outputs.get("RGB",[]))
    max_overlap=0.
    for raw in _items(foreground):
        other=_detection(raw)
        if other is None or (other["index"]==anchor["index"] and np.allclose(other["box"],box,atol=1e-12,rtol=0)):
            continue
        if other["posterior"].max()<anchor["posterior"].max():
            continue
        ob=other["box"]
        overlap=max(0.,min(box[2],ob[2])-max(box[0],ob[0]))*max(0.,min(box[3],ob[3])-max(box[1],ob[1]))/(bw*bh)
        max_overlap=max(max_overlap,overlap)
    attrs[2]=1-_clip(area*(1-min(max_overlap,.85)))

    median=calibration.get("luminance_median",calibration.get("mu_cal"))
    luminance=float(crop.mean()/255.) if crop.size else None
    if luminance is not None and _positive(median):
        attrs[3]=1. if luminance==0 else _clip(abs(log(luminance/float(median)))/log(2.5))
    else:
        audit.append("illumination_calibration_or_crop_missing")
    attrs[4]=view
    if views:
        if rgb_geometry:
            attrs[5]=_clip(1-temporal_iou)
        elif depth["depth_residual"] is not None:
            attrs[5]=_clip(.5*depth["depth_residual"]+.5*(1-temporal_iou))
        else:
            audit.append("geometry_depth_calibration_missing")
    else:
        audit.append("geometry_temporal_reference_missing")

    entropy,ring_pixels,ring_scale=_context_entropy(gray,box)
    proposal_count=sample.get("proposal_count")
    if isinstance(proposal_count,(int,np.integer)) and not isinstance(proposal_count,bool) and proposal_count>=0:
        count=int(proposal_count)-int(bool(sample.get("proposal_count_includes_target",False)))
        if count>=0 and entropy is not None:
            density=_clip((count-40)/220)
            attrs[6]=_clip(.55*density+.45*entropy)
    if attrs[6] is None:
        audit.append("context_measurement_missing")

    rgb_quality=_rgb_degradation(rgb_crop,crop)
    lap_variance=rgb_quality["laplacian_variance"]
    clipping_fraction=rgb_quality["clipping_fraction"]
    q_rgb=rgb_quality["q_rgb"]
    q_nir,nir_present=_nir_noise(_infrared_input(sample,enabled) if ("NIR" in enabled or "IR" in enabled) else None,(height,width),slices)
    if enabled-{"RGB"}:
        sensor=[]
        if q_rgb is not None:
            sensor.append(q_rgb)
        if "D" in enabled:
            sensor.append(1-depth["valid_fraction"])
        if "NIR" in enabled or "IR" in enabled:
            sensor.append(q_nir)
        attrs[7]=max(sensor) if sensor else None
    else:
        # RGB-only control disables this attribute.
        audit.append("sensing_attribute_disabled_rgb_only")

    availability=[int(x is not None and isfinite(x)) for x in attrs]
    thresholds=calibration.get("condition_thresholds",DEFAULT_CONDITION_THRESHOLDS)
    if len(thresholds)!=5 or any(not _finite_real(t) or not 0<=t<=1 for t in thresholds):
        raise ValueError("condition_thresholds require five finite values in [0,1]")
    if "condition_thresholds" not in calibration:
        audit.append("condition_thresholds_new_first_bin_choice")
    flags=[int(availability[q] and attrs[q]>=threshold) for q,threshold in zip(FAMILY_ATTRIBUTES,thresholds)]
    return {"c0":float(anchor["posterior"].max()),"sigma_b":dispersion,
            "AM_pre":float(agreement),"AM":float(agreement),"AM_fused":float(agreement),
            "Gamma_pre":geometry,"G":geometry,"S":consistency,"b_view":view,
            "rule_flags":flags,"attrs":attrs,"availability":availability,
            "qstats":{"q_rgb":q_rgb,"valid_depth_fraction":depth["valid_fraction"] if "D" in enabled else None,
                      "q_nir":q_nir if ("NIR" in enabled or "IR" in enabled) else None,
                      "mean_am_pre":float(agreement),"enabled_modalities":sorted(enabled)},
            "measurements":{"luminance":luminance,"laplacian_variance":lap_variance,
                "clipping_fraction":clipping_fraction,"depth_valid_pixels":depth["valid_pixels"],
                "depth_reliable":depth["depth_reliable"],"depth_residual":depth["depth_residual"],
                "aspect_residual":aspect,"perspective_residual":depth["perspective"],
                "temporal_iou":temporal_iou,"context_ring_pixels":ring_pixels,
                "context_ring_scale":ring_scale,"nir_present":nir_present},
            "cue_audit":audit}


def routing_vector(cues: Mapping,state: Mapping) -> list[float]:
    """Build the 17-dimensional pre-route vector; require mandatory cues."""
    s=validate_state(state)
    values=[cues.get("c0"),cues.get("sigma_b"),cues.get("AM_pre"),
            cues.get("Gamma_pre"),cues.get("b_view")]
    if any(not _finite_real(x) or not 0<=x<=1 for x in values):
        raise ValueError("routing requires finite observed/fallback pre-route cues")
    flags=cues.get("rule_flags")
    if flags is None or len(flags)!=5 or any(x not in (0,1) for x in flags):
        raise ValueError("routing requires five binary condition indicators")
    return [float(x) for x in values]+list(map(float,flags))+[s["q"],s["p"],*s["h"]]
