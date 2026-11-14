from fastapi import HTTPException
from ..models import ArchiveStudy,ClimateWindow,FeatureSpec
from ..utils.fingerprint import fingerprint
from ..utils.serialize import record
from .analysis_service import prepare_snapshot,check_multi_feasibility

_FEASIBILITY_CACHE={}
def _feasibility_key(snapshot):
    return fingerprint({"study":snapshot['study']['id'],"features":[{"code":f['code'],"position":f['position']} for f in snapshot['features']],"windows":[{"id":w['id'],"code":w['code'],"split":w['split'],"values":w['values']} for w in snapshot['windows']]})
def _multi_feasibility(snapshot):
    key=_feasibility_key(snapshot)
    if key not in _FEASIBILITY_CACHE:
        try:
            check_multi_feasibility(snapshot)
            _FEASIBILITY_CACHE[key]=(True,'train可完成BIC选群与逐群稳健拟合')
        except ValueError as e:_FEASIBILITY_CACHE[key]=(False,str(e))
        if len(_FEASIBILITY_CACHE)>16:_FEASIBILITY_CACHE.pop(next(iter(_FEASIBILITY_CACHE)))
    return _FEASIBILITY_CACHE[key]

def get_study(db,sid):
    row=db.get(ArchiveStudy,sid)
    if not row:raise HTTPException(404,"研究不存在")
    return row
def create_snapshot(db,study,mode="single"):
    if mode not in ("single","multi"):raise ValueError("运行模式仅single或multi")
    return {"study":record(study),"config":{"mode":mode},"features":[record(f) for f in db.query(FeatureSpec).filter_by(study_id=study.id).order_by(FeatureSpec.position)],"windows":[record(w) for w in db.query(ClimateWindow).filter_by(study_id=study.id).order_by(ClimateWindow.code)]}
def list_studies(db):
    out=[]
    for s in db.query(ArchiveStudy).order_by(ArchiveStudy.code):
        snapshot=create_snapshot(db,s);supported=True;reason='拆分完整、训练满秩；可运行稳健研究'
        try:prepare_snapshot(snapshot)
        except ValueError as e:supported=False;reason=str(e)
        multi_supported,multi_reason=_multi_feasibility(snapshot)
        out.append({**record(s),"counts":{role:sum(w['split']==role for w in snapshot['windows']) for role in ['train','cal','test']},"feature_count":len(snapshot['features']),"supported":supported,"reason":reason,"multi_supported":multi_supported,"multi_reason":multi_reason})
    return out
