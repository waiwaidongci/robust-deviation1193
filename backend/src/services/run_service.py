import csv,io
from redis import Redis
from fastapi import HTTPException
from ..models import DeviationRun,ReviewNote
from ..config.settings import REDIS_URL
from ..utils.serialize import record
from .study_service import get_study,create_snapshot
from .analysis_service import prepare_snapshot,check_multi_feasibility
from .audit_service import log
queue=Redis.from_url(REDIS_URL,decode_responses=True)
def _compatible(row):
    """Historical single-peak results predate mode; present them as single without rewriting JSON."""
    data=record(row)
    result=data.get('result') or {}
    if result:
        result=dict(result);result.setdefault('mode','single');data['result']=result
    data.setdefault('mode','single')
    return data
def list_runs(db):return [_compatible(r) for r in db.query(DeviationRun).order_by(DeviationRun.id.desc())]
def get_run(db,rid):
    row=db.get(DeviationRun,rid)
    if not row:raise HTTPException(404,"偏离研究不存在")
    return row
def get_run_view(db,rid):return _compatible(get_run(db,rid))
def create_run(db,data,user):
    study=get_study(db,data.study_id);snapshot=create_snapshot(db,study,data.mode)
    try:
        prepare_snapshot(snapshot)
        if data.mode=='multi':check_multi_feasibility(snapshot,data.seed)
    except ValueError as e:raise HTTPException(422,str(e))
    row=DeviationRun(**data.model_dump(),input_snapshot=snapshot,owner=user.name);db.add(row);db.flush();log(db,user.email,"deviation.enqueue",row.id);db.commit()
    try:queue.rpush('deviation:jobs',row.id)
    except Exception:row.status='failed';row.error='队列不可用，请重提';db.commit();raise HTTPException(503,row.error)
    return _compatible(row)
def export_run(db,rid):
    row=get_run(db,rid)
    if row.status!='completed' or not db.query(ReviewNote).filter_by(run_id=rid,status='approved').first():raise HTTPException(403,'仅获批完成研究可导出')
    result=row.result or {};features=result.get('features',[]);multi=result.get('mode')=='multi'
    o=io.StringIO();w=csv.writer(o)
    head=['window','split',*[f['code'] for f in features],'D_squared','cal_ge_count','upper_tail_p','preset_alpha','selected']
    if multi:head+=['group','group_cal_m','group_assignment_basis','second_group','second_group_D_squared','assignment_margin','ambiguous']
    head+=['cal_m','seed','mode','fit_hash','input_hash']
    w.writerow(head)
    for x in result.get('test',[]):
        v=x['window'];line=[v['code'],v['split'],*[v['values'][f['code']] for f in features],x['d2'],x['cal_ge_count'],x['p'],row.alpha,x['selected']]
        if multi:line+=[x['group_code'],x['group_m'],'min reweighted D2; tie->lowest group id',x['second_group_code'] or '',x['second_group_d2'] if x['second_group_d2'] is not None else '',x['assignment_margin'] if x['assignment_margin'] is not None else '',x['ambiguous']]
        line+=[result['counts']['cal'],row.seed,result.get('mode','single'),result['fit_hash'],result['input_hash']]
        w.writerow(line)
    return o.getvalue()
