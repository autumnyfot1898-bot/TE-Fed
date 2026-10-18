"""Export linked run evidence and figures from explicitly supplied observations."""
from __future__ import annotations

from collections import Counter
import csv
from hashlib import sha256
from html import escape
import json
import math
from pathlib import Path
from statistics import mean, stdev

from .analysis import read_jsonl, factorial_contrasts


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite numeric data')
    return float(value)


def _count(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return value


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def _csv(path, rows, fields):
    with Path(path).open('x', newline='', encoding='utf-8') as stream:
        writer=csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row.get(key), ensure_ascii=False, allow_nan=False)
                if isinstance(row.get(key), (dict, list)) else row.get(key) for key in fields})


def _output(path):
    result=Path(path).expanduser().resolve()
    result.mkdir(parents=True, exist_ok=False)
    return result


def _sources(paths):
    return [{'path':str(path.resolve()), 'bytes':path.stat().st_size, 'sha256':_hash(path)}
            for path in sorted(set(map(Path, paths)), key=str)]


def _manifest(out, inputs, rules, outputs):
    result={'schema_version':1, 'inputs':_sources(inputs), 'outputs':outputs,
            'rules':rules, 'missing_values':'null; no imputation',
            'output_files':[{'path':path.name, 'sha256':_hash(path), 'bytes':path.stat().st_size}
                            for path in sorted(out.iterdir()) if path.is_file()]}
    _write_json(out/'source_manifest.json', result)
    return result


def _complete_run(run):
    run=Path(run).expanduser().resolve()
    names=('configuration.json','environment.json','rounds.jsonl','metrics.json','status.json')
    paths=[run/name for name in names]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f'Completed run requires {path}')
    config, environment, rounds, metrics, status=(_json(paths[0]), _json(paths[1]),
        read_jsonl(paths[2]), _json(paths[3]), _json(paths[4]))
    total=_count(config.get('experiment',{}).get('rounds'), 'configured rounds')
    if not total or [row.get('round_index') for row in rounds] != list(range(total)):
        raise ValueError('Run rounds must be the completed sequence 0..N-1')
    if status.get('status')!='completed' or type(status.get('completed_rounds')) is not int or status.get('completed_rounds')!=total:
        raise ValueError('Run status does not commit all configured rounds')
    seed=_count(environment.get('seed'),'recorded seed')
    if 'seed' in config.get('experiment',{}) and (type(config['experiment']['seed']) is not int or config['experiment']['seed']!=seed):
        raise ValueError('Configuration seed differs from recorded seed')
    if not isinstance(metrics,dict):
        raise ValueError('Run metrics must contain named views')
    return run, config, environment, rounds, metrics, paths


def _route(route, weights, name):
    if not isinstance(route,list) or not route or len(set(route))!=len(route) \
            or any(type(head) is not int or not 0<=head<=5 for head in route):
        raise ValueError(f'{name}: invalid source route')
    if not isinstance(weights,list) or len(weights)!=len(route):
        raise ValueError(f'{name}: route/weight length mismatch')
    values=[_number(value, name) for value in weights]
    if min(values)<0 or abs(sum(values)-1)>1e-8:
        raise ValueError(f'{name}: route weights must be nonnegative and sum to one')


def audit_run(run):
    """Check identity joins and exact recorded exposure without changing a run."""
    run, config, environment, rounds, metrics, inputs=_complete_run(run)
    missing=[]; errors=[]; checks=Counter()
    def fail(message):
        errors.append(message)
    def absent(message):
        missing.append(message)
    candidates=[]
    for path in sorted(run.glob('client_*_candidate_audit.jsonl')):
        inputs.append(path); candidates.extend(read_jsonl(path))
    by_unit={}; candidate_ids=set()
    descriptors={}
    for index, row in enumerate(candidates):
        unit=(row.get('round_index'),str(row.get('client_id')))
        by_unit.setdefault(unit,[]).append(row)
        candidate_key=(*unit,row.get('local_identity'),str(row.get('candidate_id',row.get('id',row.get('anchor_id','')))))
        if candidate_key in candidate_ids:
            fail(f'duplicate candidate identity {candidate_key}')
        candidate_ids.add(candidate_key)
        if row.get('disposition') not in {'qualified','correction','suppressed'}:
            fail(f'candidate {index}: unknown disposition')
        try:
            _route(row.get('route'),row.get('weights'),f'candidate {index}')
            checks['candidate_routes']+=1
        except ValueError as exc:
            fail(str(exc))
        if row.get('descriptor_id'):
            descriptor_id=str(row['descriptor_id'])
            if descriptor_id in descriptors:
                fail(f'duplicate candidate descriptor {descriptor_id}')
            descriptors[descriptor_id]=row
            if not row.get('source_observation_key'):
                absent(f'descriptor {descriptor_id}: source_observation_key')
            else:
                payload=[row['source_observation_key'],row.get('round_index'),candidate_key[-1]]
                expected_id=sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()[:32]
                if descriptor_id!=expected_id:
                    fail(f'descriptor {descriptor_id}: candidate source/round/identity hash differs')
    for row in rounds:
        clients=row.get('clients')
        if not isinstance(clients,dict) or not clients:
            fail(f'round {row["round_index"]}: missing clients'); continue
        for client, ledger in clients.items():
            unit=(row['round_index'],str(client))
            counts=Counter(record.get('disposition') for record in by_unit.get(unit,[]))
            recorded=ledger.get('candidate_counts')
            if recorded is None:
                absent(f'round {unit[0]}/client {client}: candidate_counts')
            elif any(recorded.get(key)!=counts[key] for key in ('qualified','correction','suppressed')) \
                    or recorded.get('total')!=sum(counts.values()):
                fail(f'round {unit[0]}/client {client}: candidate counts disagree with observations')
            else:
                checks['candidate_units']+=1
    known_units={(row['round_index'],str(client)) for row in rounds for client in row.get('clients',{})}
    if set(by_unit)-known_units:
        fail('Candidate records contain a client or round outside the run')
    pool_path=run/'final.pool.json'
    pool=_json(pool_path) if pool_path.is_file() else []
    if pool_path.is_file():
        inputs.append(pool_path)
    else:
        absent('final.pool.json')
    if not isinstance(pool,list):
        raise ValueError('final.pool.json must contain a case list')
    cases={}; case_rows=[]
    for case in pool:
        uid=str(case['id'])
        if uid in cases:
            fail(f'duplicate retained case {uid}')
        cases[uid]=case
        descriptor=case.get('d',{})
        try:
            _route(descriptor.get('route'),descriptor.get('weights'),f'case {uid}')
            checks['retained_routes']+=1
        except ValueError as exc:
            fail(str(exc))
        case_rows.append({'case_id':uid,'creation_round':case.get('round'),
            'descriptor_id':descriptor.get('id'),'source_observation_key':descriptor.get('source'),
            'pattern_key':descriptor.get('pattern'),'source_route':descriptor.get('route'),
            'source_weights':descriptor.get('weights'),'fidelity':case.get('fidelity')})
    requests={}
    for path in sorted((run/'rendering').glob('round_*/*/request.json')):
        inputs.append(path); request=_json(path); uid=str(request.get('case_id'))
        if uid in requests:
            fail(f'duplicate rendered variant {uid}')
        requests[uid]=request
        descriptor=request.get('descriptor',{})
        if descriptor.get('id') not in descriptors:
            absent(f'variant {uid}: uploaded descriptor {descriptor.get("id")}')
        else:
            candidate=descriptors[descriptor['id']]
            for field,value in (('source',candidate.get('source_observation_key')),('client_id',candidate.get('client_id')),
                ('round',candidate.get('round_index')),('route',candidate.get('route')),('weights',candidate.get('weights'))):
                if descriptor.get(field)!=value:
                    fail(f'variant {uid}: descriptor {field} differs from uploaded source')
        try:
            _route(descriptor.get('route'),descriptor.get('weights'),f'variant {uid}')
        except ValueError as exc:
            fail(str(exc))
    verifications={}; fidelity_rows=[]
    for row in rounds:
        path=run/f'verification_round_{row["round_index"]:03d}.json'
        if not path.is_file():
            absent(path.name); continue
        inputs.append(path)
        for record in _json(path):
            if not record.get('case_id'):
                continue
            uid=str(record['case_id'])
            if uid in verifications:
                fail(f'duplicate case verification {uid}')
            verifications[uid]=record
            if record.get('creation_round') is None:
                absent(f'verification {uid}: creation_round')
            elif record['creation_round']!=row['round_index']:
                fail(f'verification {uid}: creation_round disagrees with file round')
            if record.get('rendered_variant_id') not in (None,uid):
                fail(f'verification {uid}: rendered_variant_id differs')
            if uid not in requests:
                absent(f'verification {uid}: rendering request')
            elif record.get('descriptor_id')!=requests[uid].get('descriptor',{}).get('id'):
                fail(f'verification {uid}: descriptor foreign key differs')
            elif uid in requests:
                descriptor=requests[uid]['descriptor']
                for field,source in (('source_observation_key','source'),('source_client_id','client_id'),
                    ('source_state_key','source_state_key'),('source_route','route'),('source_weights','weights')):
                    if field not in record:
                        absent(f'verification {uid}: {field}')
                    elif record[field]!=descriptor.get(source):
                        fail(f'verification {uid}: {field} differs from rendering source')
            if record.get('retained') is True and uid not in cases:
                fail(f'verification {uid}: retained case missing from final pool')
            audit=record.get('fidelity_audit')
            if not isinstance(audit,dict):
                absent(f'verification {uid}: fidelity_audit'); audit={}
            else:
                try:
                    _audit_fidelity(audit,record,uid,requests.get(uid,{}).get('descriptor'))
                    checks['fidelity_records']+=1
                except ValueError as exc:
                    fail(str(exc))
            fidelity_rows.append({'case_id':uid,'descriptor_id':record.get('descriptor_id'),
                'creation_round':record.get('creation_round'),'retained':record.get('retained'),
                'first_failed_gate':record.get('first_failed_gate',audit.get('first_failed_gate')),
                **{key:audit.get(key) for key in FIDELITY_FIELDS}})
    for uid,case in cases.items():
        replay=bool(config.get('replay',{}).get('enabled')) or 'replay' in rounds[0]
        if uid not in verifications:
            absent(f'retained case {uid}: verification record'+(' (frozen replay pool)' if replay else ''))
        elif verifications[uid].get('retained') is not True:
            fail(f'case {uid}: final pool contains a non-retained variant')
        elif case.get('d',{}).get('id')!=verifications[uid].get('descriptor_id'):
            fail(f'case {uid}: retained descriptor foreign key differs')
        if uid in requests:
            descriptor=case.get('d',{})
            for field in ('id','source','client_id','round','route','weights','source_state_key','pattern'):
                if descriptor.get(field)!=requests[uid].get('descriptor',{}).get(field):
                    fail(f'case {uid}: retained descriptor {field} differs from rendering request')
            if case.get('round')!=descriptor.get('round'):
                fail(f'case {uid}: retained creation round differs from descriptor round')
    selected_path=run/'selected_events.jsonl'
    events=read_jsonl(selected_path) if selected_path.is_file() else []
    if selected_path.is_file():
        inputs.append(selected_path)
    events_by_key={}; selection_rows=[]; event_ids=set()
    for event in events:
        key=(event.get('round_index'),str(event.get('client_id')),str(event.get('case_id')))
        if key in events_by_key:
            fail(f'duplicate selection {key}')
        events_by_key[key]=event
        event_id=event.get('selection_event_id')
        if event_id is None:
            absent(f'selection {key}: selection_event_id')
        elif event_id in event_ids:
            fail(f'duplicate selection_event_id {event_id}')
        else:
            identity={'client_id':str(event.get('client_id')),'round_index':event.get('round_index'),'case_id':str(event.get('case_id'))}
            expected_id=sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            if event_id!=expected_id:
                fail(f'selection {key}: event identity hash differs')
        event_ids.add(event_id)
        case=cases.get(key[2])
        if case is None:
            fail(f'selection {key}: retained case foreign key missing')
        else:
            if type(case.get('round')) is not int or case['round']>=key[0]:
                fail(f'selection {key}: same/future-round reuse')
            descriptor=case.get('d',{})
            for field,expected in (('creation_round',case.get('round')),('descriptor_id',descriptor.get('id')),
                ('source_observation_key',descriptor.get('source')),('pattern_key',descriptor.get('pattern')),
                ('source_route',descriptor.get('route')),('source_weights',descriptor.get('weights'))):
                if field not in event:
                    absent(f'selection {key}: {field}')
                elif event[field]!=expected:
                    fail(f'selection {key}: {field} differs from retained case')
        selection_rows.append({field:event.get(field) for field in SELECTION_FIELDS})
    expected_events=set(); exposure=[]
    for row in rounds:
        rid=row['round_index']
        for client,ledger in row.get('clients',{}).items():
            selected=row.get('reuse_selected_ids',{}).get(client)
            if not isinstance(selected,list) or len(set(selected))!=len(selected):
                fail(f'round {rid}/client {client}: unique selected IDs required'); continue
            expected_events.update((rid,str(client),str(uid)) for uid in selected)
            try:
                attempted=_count(ledger.get('attempted_optimizer_steps'),'attempted_optimizer_steps')
                applied=_count(ledger.get('optimizer_steps'),'optimizer_steps')
                skipped=_count(ledger.get('amp_skipped_steps'),'amp_skipped_steps')
                if attempted!=applied+skipped:
                    raise ValueError('applied updates plus AMP skips differ from attempts')
                tracked=ledger.get('case_presentations')
                if not isinstance(tracked,dict):
                    absent(f'round {rid}/client {client}: case_presentations'); continue
                if set(tracked)!=set(map(str,selected)):
                    raise ValueError('case presentation IDs differ from selected IDs')
                presentations=0; applied_ids=set(); skipped_ids=set()
                for uid,values in tracked.items():
                    count=_count(values.get('presentations'),'case presentations')
                    supervised=_count(values.get('supervised_presentations'),'supervised presentations')
                    if supervised>count:
                        raise ValueError('supervised presentations exceed actual presentations')
                    if type(values.get('detection_supervision_enabled')) is not bool:
                        raise ValueError('detection supervision flag must be explicit')
                    if not values['detection_supervision_enabled'] and supervised:
                        raise ValueError('disabled detection supervision has supervised presentations')
                    step_sets=[]
                    for field in ('attempted_step_ids','applied_step_ids','amp_skipped_step_ids'):
                        ids=values.get(field)
                        if not isinstance(ids,list) or len(set(ids))!=len(ids) \
                                or any(type(step) is not int or not 0<=step<attempted for step in ids):
                            raise ValueError(f'case {uid}: invalid {field}')
                        step_sets.append(set(ids))
                    if step_sets[1]&step_sets[2] or step_sets[0]!=step_sets[1]|step_sets[2]:
                        raise ValueError(f'case {uid}: step outcomes do not partition attempted steps')
                    if len(step_sets[0])>count or len(step_sets[1])>applied or len(step_sets[2])>skipped:
                        raise ValueError(f'case {uid}: steps exceed actual unit totals')
                    applied_ids.update(step_sets[1]); skipped_ids.update(step_sets[2])
                    event=events_by_key.get((rid,str(client),str(uid)))
                    if event:
                        for field in ('supervised_presentations','detection_supervision_enabled',
                            'attempted_step_ids','applied_step_ids','amp_skipped_step_ids'):
                            if event.get(field)!=values.get(field):
                                raise ValueError(f'case {uid}: selection and training {field} differ')
                        if event.get('actual_presentations')!=count:
                            raise ValueError(f'case {uid}: selection and training actual presentations differ')
                    presentations+=count
                if applied_ids&skipped_ids:
                    raise ValueError('a shared optimizer step has conflicting case outcomes')
                if len(applied_ids)>applied or len(skipped_ids)>skipped:
                    raise ValueError('distinct case step outcomes exceed client update totals')
                if presentations!=ledger.get('presentations',{}).get('correction'):
                    raise ValueError('sum of per-case presentations differs from correction total')
                checks['exposure_units']+=1
                exposure.append({'round_index':rid,'client_id':str(client),'selected_cases':len(selected),
                    'actual_presentations':presentations,'attempted_optimizer_steps':attempted,
                    'applied_optimizer_steps':applied,'amp_skipped_steps':skipped})
            except ValueError as exc:
                fail(f'round {rid}/client {client}: {exc}')
    if expected_events!=set(events_by_key):
        fail('Selection event identities differ from the round selection ledger')
    report={'schema_version':1,'status':'invalid' if errors else 'incomplete' if missing else 'passed',
        'run':str(run),'method':config.get('experiment',{}).get('method'),'seed':environment.get('seed'),
        'rounds':len(rounds),'checks':dict(checks),'errors':errors,'missing':sorted(set(missing)),
        'counts':{'candidates':len(candidates),'uploaded_descriptors':len(descriptors),
            'rendered_variants':len(requests),'retained_cases':len(cases),'selection_events':len(events)},
        'exposure':exposure,'case_rows':case_rows,'selection_rows':selection_rows,'fidelity_rows':fidelity_rows}
    return report, inputs


FIDELITY_FIELDS=('schema_version','rule_version','attribute_names','descriptor_attrs','requested_intervals',
    'measured_severities','measurement_status','active_mask','weights','normalizers','attribute_fidelities',
    'active_weight_mass','evaluated','valid','score','reason')
SELECTION_FIELDS=('selection_event_id','client_id','round_index','case_id','descriptor_id','source_observation_key',
    'pattern_key','creation_round','source_route','source_weights','actual_presentations','supervised_presentations',
    'detection_supervision_enabled','attempted_step_ids','applied_step_ids','amp_skipped_step_ids')


def _audit_fidelity(audit, record, uid, descriptor=None):
    from . import core
    if audit.get('schema_version')!=1 or audit.get('rule_version')!='tefed-fidelity-v1':
        raise ValueError(f'verification {uid}: fidelity rule version differs')
    if audit.get('attribute_names')!=list(core.ATTRIBUTES) or audit.get('weights')!=list(core.FIDELITY_WEIGHTS) \
            or audit.get('normalizers')!=list(core.FIDELITY_TOLERANCES):
        raise ValueError(f'verification {uid}: fixed fidelity definitions differ')
    if audit.get('first_failed_gate')!=record.get('first_failed_gate'):
        raise ValueError(f'verification {uid}: first failed gate differs')
    if descriptor is not None:
        descriptor=core.validate_descriptor(descriptor)
        if audit.get('descriptor_attrs')!=descriptor['attrs'] or audit.get('active_mask')!=descriptor['mask'] \
                or audit.get('requested_intervals')!=core.descriptor_intervals(descriptor['attrs']):
            raise ValueError(f'verification {uid}: fidelity descriptor inputs differ from rendering request')
    for field in ('active_mask','weights','normalizers','requested_intervals','attribute_fidelities'):
        if not isinstance(audit.get(field),list) or len(audit[field])!=8:
            raise ValueError(f'verification {uid}: {field} must contain eight entries')
    if any(type(flag) not in (int,bool) or flag not in (0,1) for flag in audit['active_mask']):
        raise ValueError(f'verification {uid}: active mask must be binary')
    weights=[_number(value,'fidelity weight') for value in audit['weights']]
    mass=sum(weight for weight,active in zip(weights,audit['active_mask']) if active)
    if min(weights)<0 or abs(mass-_number(audit.get('active_weight_mass'),'active mass'))>1e-10:
        raise ValueError(f'verification {uid}: masked fidelity mass differs')
    if audit.get('evaluated') is False:
        if audit.get('score') is not None:
            raise ValueError(f'verification {uid}: unevaluated fidelity has a score')
        return
    if audit.get('evaluated') is not True:
        raise ValueError(f'verification {uid}: evaluated flag missing')
    if audit.get('valid') is True:
        if mass<=0:
            raise ValueError(f'verification {uid}: valid fidelity has empty mask')
        measured=audit.get('measured_severities')
        if not isinstance(measured,list) or len(measured)!=8:
            raise ValueError(f'verification {uid}: measured severities missing')
        for active,interval,x,normalizer,value in zip(audit['active_mask'],audit['requested_intervals'],
                measured,audit['normalizers'],audit['attribute_fidelities']):
            if not active:
                if value is not None:
                    raise ValueError(f'verification {uid}: inactive attribute has a fidelity score')
                continue
            if not isinstance(interval,list) or len(interval)!=2:
                raise ValueError(f'verification {uid}: requested interval invalid')
            low,high=(_number(bound,'interval bound') for bound in interval)
            x=_number(x,'measured severity'); tolerance=_number(normalizer,'normalizer')
            if not 0<=low<=high<=1 or not 0<=x<=1 or tolerance<=0:
                raise ValueError(f'verification {uid}: interval, severity or normalizer invalid')
            expected=max(0.,1-max(low-x,0.,x-high)/tolerance)
            if abs(expected-_number(value,'attribute fidelity'))>1e-10:
                raise ValueError(f'verification {uid}: attribute fidelity differs from interval rule')
        score=sum(weight*_number(value,'active attribute fidelity') for weight,value,active in
            zip(weights,audit['attribute_fidelities'],audit['active_mask']) if active)/mass
        if abs(score-_number(audit.get('score'),'fidelity score'))>1e-10 \
                or abs(score-_number(record.get('fidelity'),'verification fidelity'))>1e-10:
            raise ValueError(f'verification {uid}: masked fidelity score differs')
    elif audit.get('score') is not None:
        raise ValueError(f'verification {uid}: invalid fidelity has a score')


def export_audit(run, output):
    report,inputs=audit_run(run)
    out=_output(output)
    tables={name:report.pop(name+'_rows') for name in ('case','selection','fidelity')}
    _write_json(out/'audit.json',report)
    _csv(out/'cases.csv',tables['case'],('case_id','creation_round','descriptor_id','source_observation_key',
        'pattern_key','source_route','source_weights','fidelity'))
    _csv(out/'selections.csv',tables['selection'],SELECTION_FIELDS)
    _csv(out/'fidelity.csv',tables['fidelity'],('case_id','descriptor_id','creation_round','retained','first_failed_gate',*FIDELITY_FIELDS))
    _manifest(out,inputs,{
        'candidate_counts':'Actual observations grouped by round/client and disposition; compared to candidate_counts.',
        'foreign_keys':'candidate descriptor -> rendering request -> verification -> final.pool -> selected event.',
        'reuse':'retained creation_round < selection round; each client/round/case selects once.',
        'fidelity':'fixed active mask; sum(weight*attribute_fidelity)/active_weight_mass; unevaluated remains null.',
        'exposure':'exact per-case presentation sum; attempted steps partition into applied or AMP-skipped IDs.'},
        {'audit.json':'status/rounds/counts/checks/errors/missing/exposure from named run ledgers',
         'cases.csv':'final.pool.json: id/round/d/fidelity',
         'selections.csv':'selected_events.jsonl: identity/source/route/presentation/step fields',
         'fidelity.csv':'verification_round_*.json: case/descriptor/retained and fidelity_audit fields'})
    return report


COLORS=('#165d9c','#d16728','#32865c','#8b55a5','#ad3946','#617284')


def _svg_lines(path, title, series, *, y_label):
    """Draw explicit numeric series; each series retains its measured x positions."""
    points=[point for item in series for point in item['points']]
    if not points:
        raise ValueError(f'No measured points for {title}')
    xmin=min(x for x,_ in points); xmax=max(x for x,_ in points)
    ymin=min(y for _,y in points); ymax=max(y for _,y in points)
    if xmin==xmax:
        xmin-=.5; xmax+=.5
    delta=(ymax-ymin)*.08 or max(abs(ymin)*.1,.1)
    ymin-=delta; ymax+=delta
    width=1020; height=460+22*len(series)
    def xy(x,y):
        return 80+(x-xmin)/(xmax-xmin)*880,390-(y-ymin)/(ymax-ymin)*320
    content=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="80" y="30" font-family="sans-serif" font-size="20">{escape(title)}</text>',
        '<path d="M80 60V390H960" fill="none" stroke="#444"/>',
        f'<text x="80" y="48" font-family="sans-serif" font-size="12">{escape(y_label)}</text>',
        '<text x="475" y="432" font-family="sans-serif" font-size="13">Completed round (1-based)</text>']
    for i in range(6):
        value=ymin+(ymax-ymin)*i/5; _,y=xy(xmin,value)
        content.append(f'<text x="72" y="{y+4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.4g}</text>')
    for value in sorted({x for x,_ in points}):
        x,_=xy(value,ymin)
        content.append(f'<text x="{x:.2f}" y="409" text-anchor="middle" font-family="sans-serif" font-size="11">{value:g}</text>')
    for i,item in enumerate(series):
        color=COLORS[i%len(COLORS)]
        coords=' '.join(f'{xy(x,y)[0]:.2f},{xy(x,y)[1]:.2f}' for x,y in item['points'])
        content.append(f'<polyline points="{coords}" stroke="{color}" stroke-width="2" fill="none"/>')
        for x,y in item['points']:
            x,y=xy(x,y); content.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}"/>')
        content.append(f'<text x="80" y="{460+i*22}" fill="{color}" font-family="sans-serif" font-size="12">{escape(item["label"])}</text>')
    content.append('</svg>'); path.write_text('\n'.join(content),encoding='utf-8')


def _svg_bars(path,title,rows):
    if not rows:
        raise ValueError(f'No measured values for {title}')
    width=max(760,145*len(rows)+110); height=520
    low=min(0.,min(row['mean']-(row.get('sample_sd') or 0) for row in rows))
    high=max(0.,max(row['mean']+(row.get('sample_sd') or 0) for row in rows))
    if high==low:
        high=low+1
    def py(value):
        return 360-(value-low)/(high-low)*280
    content=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="55" y="32" font-family="sans-serif" font-size="20">{escape(title)}</text>',
        f'<path d="M65 65V360H{width-25}" fill="none" stroke="#444"/>',
        f'<path d="M65 {py(0):.2f}H{width-25}" fill="none" stroke="#bbb"/>']
    for tick in range(6):
        value=low+(high-low)*tick/5
        content.append(f'<text x="60" y="{py(value)+4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.4g}</text>')
    span=(width-100)/len(rows)
    for i,row in enumerate(rows):
        x=80+i*span; y=py(row['mean']); zero=py(0); center=x+span*.3
        content.append(f'<rect x="{x:.2f}" y="{min(y,zero):.2f}" width="{span*.6:.2f}" height="{abs(zero-y):.2f}" fill="{COLORS[i%len(COLORS)]}"/>')
        sd=row.get('sample_sd')
        if sd is not None:
            top,bottom=py(row['mean']+sd),py(row['mean']-sd)
            content.append(f'<path d="M{center:.2f} {top:.2f}V{bottom:.2f}M{center-5:.2f} {top:.2f}H{center+5:.2f}M{center-5:.2f} {bottom:.2f}H{center+5:.2f}" stroke="#222"/>')
        for j,part in enumerate(row['label'].split(' / ')):
            content.append(f'<text x="{center:.2f}" y="{386+j*17}" text-anchor="middle" font-family="sans-serif" font-size="11">{escape(part)}</text>')
        annotation=f'{row["mean"]:.4g}; n={row["n"]}'+(' SD unavailable' if sd is None else f'; SD={sd:.3g}')
        content.append(f'<text x="{center:.2f}" y="{y-8 if row["mean"]>=0 else y+16:.2f}" text-anchor="middle" font-family="sans-serif" font-size="10">{escape(annotation)}</text>')
    content.append('</svg>'); path.write_text('\n'.join(content),encoding='utf-8')


def plot_runs(runs, output, *, factorial_records=None, seeds=None):
    """Plot actual per-client losses and seed metrics; no interpolated observations."""
    paths=[Path(run).expanduser().resolve() for run in runs]
    if not paths or len(set(paths))!=len(paths):
        raise ValueError('Supply distinct completed run directories')
    inputs=[]; series=[]; values=[]; groups={}; seen=set(); identities=set(); methods={}; shared=None
    for path in paths:
        run,config,environment,rounds,metrics,files=_complete_run(path); inputs.extend(files)
        method=str(config.get('experiment',{}).get('method'))
        seed=environment.get('seed'); identity=(method,str(seed))
        if identity in seen:
            raise ValueError(f'Duplicate method/seed {identity}')
        seen.add(identity)
        data_hash=environment.get('manifest_sha256')
        if not isinstance(data_hash,str) or len(data_hash)!=64:
            raise ValueError('Metric comparison requires recorded manifest_sha256')
        identities.add(data_hash)
        from .analysis import _run_protocol
        protocol=_run_protocol(config,environment,run)
        if (run/'calibration.json').is_file():
            inputs.append(run/'calibration.json')
        common=json.loads(json.dumps(protocol))
        from .offline import METHOD_SECTIONS
        for section in METHOD_SECTIONS:
            common['configuration'].pop(section,None)
        common['configuration']['experiment'].pop('method',None)
        if shared is not None and common!=shared:
            raise ValueError('Compared methods must share data, calibration, initialization, model, evaluation and training settings')
        shared=common
        protocol_hash=sha256(json.dumps(protocol,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        if method in methods and methods[method]!=protocol_hash:
            raise ValueError('Within-method seed runs must share one configuration and input protocol')
        methods[method]=protocol_hash
        clients={str(client) for row in rounds for client in row['clients']}
        for client in sorted(clients):
            points=[]
            for row in rounds:
                if client not in row['clients']:
                    raise ValueError('Training curves require stable client identities across rounds')
                value=row['clients'][client].get('mean_step_loss')
                if value is not None:
                    value=_number(value,'mean_step_loss'); points.append((row['round_index']+1,value))
                    values.append({'kind':'training','run':str(run),'method':method,'seed':seed,'client_id':client,
                        'round':row['round_index']+1,'view':None,'metric':'mean_step_loss','value':value})
            if points:
                series.append({'label':f'{method}; seed={seed}; client={client}','points':points})
        for view,scores in metrics.items():
            for metric in ('mAP50','AP75','mAP50_95'):
                value=_number(scores.get(metric),f'{view}/{metric}')
                groups.setdefault((method,view,metric),[]).append((seed,value))
                values.append({'kind':'final_metric','run':str(run),'method':method,'seed':seed,
                    'client_id':None,'round':len(rounds),'view':view,'metric':metric,'value':value})
    if len(identities)!=1:
        raise ValueError('Compared runs must share the recorded evaluation data manifest')
    declared=None
    if seeds is not None:
        declared=list(seeds)
        if not declared or len(set(declared))!=len(declared) or any(type(seed) is not int or seed<0 for seed in declared):
            raise ValueError('Declared seeds must be distinct nonnegative integers')
    bars=[]
    for (method,view,metric),rows in sorted(groups.items()):
        if declared is not None and {str(seed) for seed,_ in rows}!=set(map(str,declared)):
            raise ValueError(f'Missing declared seed for {method}/{view}/{metric}')
        scores=[value for _,value in rows]
        bars.append({'label':f'{method} / {view} / {metric}','mean':mean(scores),
            'sample_sd':stdev(scores) if len(scores)>1 else None,'n':len(scores)})
    factors=None
    if factorial_records is not None:
        if declared is None or len(declared)<2:
            raise ValueError('Factorial effects require at least two declared actual seeds')
        path=Path(factorial_records).resolve(); records=read_jsonl(path); inputs.append(path)
        hashes={row.get('protocol_sha256') for row in records}
        if len(hashes)!=1 or None in hashes:
            raise ValueError('Factorial records must share one recorded execution protocol')
        inputs.extend(_factorial_sources(path,records))
        factors=factorial_contrasts(records,expected_seeds=declared)
    out=_output(output)
    if series:
        _svg_lines(out/'training_curves.svg','Measured training loss by client',series,y_label='Mean optimizer step loss')
    if not bars:
        raise ValueError('No final measured metrics')
    _svg_bars(out/'metric_comparison.svg','Final detection metrics (0-1); sample SD across actual seeds',bars)
    if factors:
        _svg_bars(out/'factorial_effects.svg','Factorial signed contrasts; sample SD across declared seeds',
            [{'label':key,'mean':entry['mean'],'sample_sd':entry['sample_sd'],'n':entry['n']}
             for key,entry in factors['summary'].items()])
        _write_json(out/'factorial_effects.json',factors)
    _csv(out/'plot_values.csv',values,('kind','run','method','seed','client_id','round','view','metric','value'))
    manifest=_manifest(out,inputs,{
        'training_curves.svg':'Each point is clients[client].mean_step_loss at round_index+1; missing points omitted, never interpolated.',
        'metric_comparison.svg':'metrics[view][metric]; arithmetic seed mean, sample SD only for n>=2; metric values remain 0-1.',
        'factorial_effects.svg':'Supplied A/B/C values; S8 signed sums scaled 1/4,1/2,1; no favorable-sign reversal.'},
        {'plot_values.csv':'rounds.jsonl client mean_step_loss and metrics.json view scores',
         'training_curves.svg':'plot_values.csv kind=training; individual clients and seeds',
         'metric_comparison.svg':'plot_values.csv kind=final_metric grouped by method/view/metric',
         'factorial_effects.svg':'explicit factorial_records input only; complete eight cells per declared seed'})
    _csv(out/'source_manifest.csv',manifest['inputs'],('path','bytes','sha256'))
    return {'output':str(out),'series':len(series),'metric_groups':len(bars),'factorial':factors is not None}


def _factorial_sources(path, records):
    """Join exported values to the completed jobs that produced them."""
    name=path.stem
    metric=next((metric for metric in ('mAP50_95','mAP50','AP75') if name.endswith('_'+metric)),None)
    if metric is None or not name.startswith('factorial_'):
        raise ValueError('Use the executed suite factorial_<view>_<metric>.jsonl export')
    view=name[len('factorial_'):-(len(metric)+1)]
    index=path.parent/'factorial_runs.jsonl'; plan_path=path.parent/'plan.json'; execution_path=path.parent/'execution.json'
    for file in (index,plan_path,execution_path):
        if not file.is_file():
            raise FileNotFoundError(f'Factorial figure requires executed suite evidence: {file}')
    plan=_json(plan_path); execution=_json(execution_path)
    if plan.get('suite_plan')!='factorial' or execution.get('status')!='completed' \
            or execution.get('suite_sha256')!=plan.get('suite_sha256'):
        raise ValueError('Factorial suite execution is not completed or differs from its plan')
    indexed=read_jsonl(index); expected={}; inputs=[index,plan_path,execution_path]
    jobs=plan.get('jobs',[]); entries=execution.get('jobs',[])
    if not jobs or len(entries)!=len(jobs) or len(indexed)!=len(jobs):
        raise ValueError('Factorial suite job/index membership differs')
    indexed_by_key={}
    for row in indexed:
        key=(row.get('seed'),tuple(row.get('factors',{}).get(axis) for axis in ('A','B','C')))
        if key in indexed_by_key:
            raise ValueError('Duplicate factorial run identity')
        indexed_by_key[key]=row
    for job,entry in zip(jobs,entries):
        if entry.get('status')!='completed' or any(entry.get(key)!=job.get(key) for key in ('variant','seed','output')):
            raise ValueError('Factorial execution job differs from plan or is unfinished')
        key=(job.get('seed'),tuple(job.get('factors',{}).get(axis) for axis in ('A','B','C')))
        row=indexed_by_key.get(key)
        if row is None or row.get('status')!='completed' or row.get('output')!=job.get('output') \
                or row.get('protocol_sha256')!=plan.get('protocol_sha256'):
            raise ValueError('Factorial exported identity differs from executed job')
        _,config,environment,_,metrics,files=_complete_run(job['output']); inputs.extend(files)
        if config.get('factorial',{}).get('factors')!=job.get('factors') or environment['seed']!=job.get('seed') \
                or metrics!=row.get('metrics') or metrics!=entry.get('result',{}).get('metrics'):
            raise ValueError('Factorial execution configuration or actual metrics differ')
        expected[key]=_number(metrics.get(view,{}).get(metric),'factorial actual metric')
    observed={}
    for row in records:
        key=(row.get('seed'),tuple(row.get('factors',{}).get(axis) for axis in ('A','B','C')))
        if key in observed or row.get('protocol_sha256')!=plan.get('protocol_sha256'):
            raise ValueError('Duplicate factorial value or protocol differs')
        observed[key]=_number(row.get('value'),'factorial value')
    if observed!=expected:
        raise ValueError('Factorial values differ from completed run metrics')
    return inputs


def qualitative_comparison(manifest_path, view, ids, predictions, output, *, image_size, threshold, view_spec=None):
    """Render fixed identities with one declared letterbox and confidence threshold."""
    from PIL import Image, ImageDraw
    import numpy as np
    from .data import load_manifest, load_sample, apply_unseen, parse_annotation

    manifest_path=Path(manifest_path).resolve()
    manifest=load_manifest(manifest_path,strict_files=False)
    if view not in {'participating','unseen','new-client','hard'}:
        raise ValueError('Choose an explicit participating/unseen/new-client/hard view')
    views={str(name).lower().replace('_','-'):members for name,members in manifest.metadata.get('views',{}).items()}
    if view not in views:
        raise ValueError('Qualitative comparison requires explicit manifest view membership')
    members=set(map(str,views[view])); ids=list(map(str,ids))
    if not ids or len(set(ids))!=len(ids) or not set(ids)<=members:
        raise ValueError('Fixed identities must be unique members of the declared view')
    if not isinstance(predictions,dict) or len(predictions)<2 or any(not isinstance(label,str) or not label.strip() for label in predictions):
        raise ValueError('Supply at least two distinct method labels and prediction files')
    if any(label.casefold() in {'rgb','depth','nir','ground truth'} for label in predictions):
        raise ValueError('Method labels must differ from modality and Ground truth panel names')
    if len({Path(path).expanduser().resolve() for path in predictions.values()})!=len(predictions):
        raise ValueError('Method labels require distinct prediction files')
    if type(image_size) is not int or not 32<=image_size<=4096:
        raise ValueError('image_size must be an explicit integer in [32,4096]')
    threshold=_number(threshold,'confidence threshold')
    if not 0<=threshold<=1:
        raise ValueError('confidence threshold must be in [0,1]')
    inputs=[manifest_path]; spec=None
    if view_spec is not None:
        view_spec=Path(view_spec).resolve(); inputs.append(view_spec); spec=_json(view_spec)
        if spec.get('view')!=view:
            raise ValueError('View transformation specification names a different view')
    if view=='unseen' and (not isinstance(spec,dict) or not isinstance(spec.get('perturbation'),dict)
            or type(spec.get('seed')) is not int):
        raise ValueError('Unseen view requires explicit perturbation and seed in --view-spec')
    if view!='unseen' and spec and spec.get('perturbation'):
        raise ValueError('Only the unseen view accepts perturbations')
    rows_by_method={}; prediction_info=[]; prediction_evidence={}
    for label,raw in predictions.items():
        path=Path(raw).resolve(); inputs.append(path); rows=read_jsonl(path); indexed={}
        for row in rows:
            uid=str(row.get('id'))
            if uid in indexed:
                raise ValueError(f'{label}: duplicate prediction identity {uid}')
            if row.get('view')!=view or row.get('coordinate_system')!='letterbox-normalized-xyxy':
                raise ValueError(f'{label}: prediction view or letterbox coordinate declaration differs')
            if not isinstance(row.get('predictions'),list):
                raise ValueError(f'{label}: predictions must be an explicit list')
            for detection in row['predictions']:
                parse_annotation(detection)
                score=_number(detection.get('score'),'prediction score')
                if not 0<=score<=1:
                    raise ValueError('Prediction score must be in [0,1]')
            indexed[uid]=row
        if set(indexed)!=members:
            raise ValueError(f'{label}: prediction IDs must exactly match the declared view')
        rows_by_method[label]=indexed
        sidecar=path.with_suffix(path.suffix+'.manifest.json')
        if not sidecar.is_file():
            sidecar=path.parent/'evaluation_manifest.json'
        evidence=_json(sidecar) if sidecar.is_file() else None
        if evidence is not None:
            inputs.append(sidecar)
            from .evaluation_manifest import identity_sha256, record_metadata
            if evidence.get('coordinate_system')!='letterbox-normalized-xyxy' or evidence.get('schema_version')!=1:
                raise ValueError(f'{label}: evaluation manifest coordinate/schema differs')
            if evidence.get('manifest_sha256')!=identity_sha256(manifest.as_dict()):
                raise ValueError(f'{label}: evaluation manifest dataset identity differs')
            view_evidence=evidence.get('views',{}).get(view)
            if not isinstance(view_evidence,dict) or not isinstance(view_evidence.get('records'),list):
                raise ValueError(f'{label}: evaluation manifest view records required')
            protocol=view_evidence.get('transform',{})
            if protocol.get('image_size') not in (image_size,[image_size,image_size]):
                raise ValueError(f'{label}: evaluation manifest image size differs')
            if view=='unseen' and (protocol.get('seed')!=spec['seed'] or protocol.get('unseen')!=spec['perturbation']):
                raise ValueError(f'{label}: evaluation manifest unseen transformation differs')
            if view!='unseen' and protocol.get('unseen') is not None:
                raise ValueError(f'{label}: unexpected unseen transformation in declared view')
            records=view_evidence['records']; observed={str(row.get('id')):row for row in records}
            if len(observed)!=len(records) or set(observed)!=members:
                raise ValueError(f'{label}: evaluation manifest identities differ')
            if any(any(row.get(key)!=value for key,value in record_metadata(manifest.by_id[uid]).items()) for uid,row in observed.items()):
                raise ValueError(f'{label}: evaluation manifest observation metadata differs')
            if not isinstance(evidence.get('model_identity'),dict) or not evidence['model_identity'].get('sha256'):
                raise ValueError(f'{label}: evaluation manifest model identity required')
            recorded_file=evidence.get('prediction_files',{}).get(view) if isinstance(evidence.get('prediction_files'),dict) else None
            if not isinstance(recorded_file,dict) or recorded_file.get('sha256')!=_hash(path) \
                    or recorded_file.get('bytes')!=path.stat().st_size:
                raise ValueError(f'{label}: evaluation manifest prediction file identity differs')
            protocol_sha=sha256(json.dumps(protocol,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            if any(row.get('transform_sha256')!=protocol_sha for row in records):
                raise ValueError(f'{label}: evaluation transform protocol hash differs')
            prediction_evidence[label]={'records':observed,'protocol':protocol}
        else:
            prediction_evidence[label]=None
        prediction_info.append({'method':label,'path':str(path),'sha256':_hash(path),
            'checkpoint':{uid:indexed[uid].get('checkpoint') for uid in ids},
            'checkpoint_sha256':{uid:indexed[uid].get('checkpoint_sha256') for uid in ids},
            'model_identity':evidence.get('model_identity') if evidence else None,
            'evaluation_manifest':str(sidecar) if evidence else None,
            'input_identity':'exact prediction file SHA-256; checkpoint remains null when unrecorded'})
    loaded=[]; selection=[]
    for uid in ids:
        record=manifest.by_id[uid]
        if record.split not in {'test','hard'}:
            raise ValueError('Fixed qualitative identities require explicit held-out ground truth')
        for asset in (record.rgb,record.depth,record.nir):
            if asset is not None:
                if not asset.is_file():
                    raise FileNotFoundError(asset)
                inputs.append(asset)
        sample=load_sample(record,size=image_size,include_labels=True)
        if view=='unseen':
            sample=apply_unseen(sample,identity=uid,perturbation=spec['perturbation'],seed=spec['seed'])
        transform_hash=sha256(json.dumps(sample['meta'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        for label,indexed in rows_by_method.items():
            evidence=prediction_evidence[label]
            if evidence:
                record_evidence=evidence['records'][uid]
                geometry={key:sample['meta'][key] for key in ('original_size','size','resized_size','scale','scale_xy','pad')}
                if record_evidence.get('letterbox')!=json.loads(json.dumps(geometry)):
                    raise ValueError(f'{label}/{uid}: recorded letterbox geometry differs')
                if record_evidence.get('applied_unseen')!=sample['meta'].get('unseen'):
                    raise ValueError(f'{label}/{uid}: recorded actual unseen transformation differs')
            else:
                row=indexed[uid]
                if row.get('image_size') not in (image_size,[image_size,image_size]) \
                        or row.get('transform_hash_kind')!='sample-metadata-sha256-v1' \
                        or row.get('transform_sha256')!=transform_hash:
                    raise ValueError(f'{label}/{uid}: evaluation manifest or explicit image_size and sample-metadata transform identity required')
        loaded.append(sample)
        selection.append({'id':uid,'view':view,'client_id':record.client_id,'split':record.split,
            'transform':sample['meta'],'transform_sha256':transform_hash,'targets':sample['targets'],
            'condition_tag':record.metadata.get('condition_tag'),'condition_tags':record.metadata.get('condition_tags'),
            'tag_definition_version':record.metadata.get('tag_definition_version'),
            'condition_tags_version':record.metadata.get('condition_tags_version'),
            'paired_group_id':record.metadata.get('paired_group_id')})
    modalities=['rgb']+[name for name in ('depth','nir') if any(sample.get(name) is not None for sample in loaded)]
    labels=modalities+['Ground truth']+list(rows_by_method)
    cell=image_size; header=34; row_height=cell+header
    sheet=Image.new('RGB',(len(labels)*cell,len(loaded)*row_height),(255,255,255)); draw=ImageDraw.Draw(sheet)
    depth_ranges={}
    for row_index,(uid,sample) in enumerate(zip(ids,loaded)):
        rgb=Image.fromarray(sample['rgb']).convert('RGB')
        for column,label in enumerate(labels):
            image=rgb.copy(); y=row_index*row_height+header; x=column*cell
            if label=='nir':
                image=Image.fromarray(sample['nir']).convert('RGB') if sample.get('nir') is not None else Image.new('RGB',(cell,cell),(230,230,230))
            elif label=='depth':
                depth=sample.get('depth')
                if depth is None:
                    image=Image.new('RGB',(cell,cell),(230,230,230)); depth_ranges[uid]=None
                else:
                    valid=depth[depth>0]; high=float(valid.max()) if valid.size else 0.
                    scaled=np.rint(np.clip(depth/high,0,1)*255).astype(np.uint8) if high else np.zeros_like(depth,dtype=np.uint8)
                    image=Image.fromarray(scaled).convert('RGB'); depth_ranges[uid]={'min':0.,'max':high,'unit':'recorded sensor units'}
            if label=='Ground truth' or label in rows_by_method:
                detections=sample['targets'] if label=='Ground truth' else rows_by_method[label][uid]['predictions']
                pen=ImageDraw.Draw(image)
                for detection in detections:
                    if label!='Ground truth' and detection['score']<threshold:
                        continue
                    box=[float(value)*cell for value in detection['box']]
                    color=(20,155,65) if label=='Ground truth' else (230,70,30)
                    pen.rectangle(box,outline=color,width=max(1,cell//240))
                    text=f'{detection["class_id"]}'+('' if label=='Ground truth' else f' {detection["score"]:.3f}')
                    pen.text((box[0],max(0,box[1]-12)),text,fill=color)
            sheet.paste(image,(x,y)); draw.text((x+4,y-header+4),f'{uid} | {label}',fill=(20,20,20))
            if label in ('depth','nir') and sample.get(label) is None:
                draw.text((x+4,y+4),'Unavailable',fill=(70,70,70))
    out=_output(output); sheet.save(out/'qualitative.png')
    _write_json(out/'selection_manifest.json',{'schema_version':1,'selection':'explicit user-supplied IDs; no quality ranking',
        'ids':ids,'view':view,'image_size':[cell,cell],'coordinate_system':'letterbox-normalized-xyxy',
        'confidence_threshold':threshold,'threshold_operator':'>=','postprocessing':'none; supplied prediction records',
        'view_spec':spec,'predictions':prediction_info,'records':selection,'depth_display_ranges':depth_ranges})
    provenance=_manifest(out,inputs,{
        'qualitative.png':'Fixed declared identities; data.load_sample shared letterbox, optional explicit unseen transform; native GT boxes and supplied method boxes.',
        'depth_display':'Linear grayscale from zero to per-identity positive maximum; display only, range recorded.',
        'confidence':'Same >= threshold for every supplied method; no added NMS or selected best cases.'},
        {'qualitative.png':'manifest records RGB/depth/NIR/labels and labelled prediction JSONL rows',
         'selection_manifest.json':'explicit IDs/view/threshold/size and observed transformation/asset/prediction identities; absent condition/checkpoint fields null'})
    _csv(out/'source_manifest.csv',provenance['inputs'],('path','bytes','sha256'))
    return {'output':str(out),'ids':ids,'view':view,'methods':list(rows_by_method)}
