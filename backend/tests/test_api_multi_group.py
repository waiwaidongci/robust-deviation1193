import os,sys
import unittest
os.environ['DATABASE_URL']='sqlite:////tmp/devtest-unittest.db'
os.environ['REDIS_URL']='redis://localhost:6399/0'
PATH='/tmp/devtest-unittest.db'
if os.path.exists(PATH):os.remove(PATH)
sys.path.insert(0,'/workspace/backend')
sys.path.insert(0,'/workspace')
from sqlalchemy import create_engine,text
from sqlalchemy.orm import sessionmaker
import src.config.database as database
engine=create_engine(os.environ['DATABASE_URL'],connect_args={'check_same_thread':False})
database.engine=engine;database.SessionLocal=sessionmaker(bind=engine,expire_on_commit=False)
from fastapi.testclient import TestClient
import src.main as main
main.engine=engine
main.queue.ping=lambda:True
main.queue.rpush=lambda *a:1
from src.services.analysis_service import analyze_deviation
from src.models import DeviationRun
from scripts.seed_demo import seed

class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed();cls.client=TestClient(main.app);cls.tokens={}
        for role in ('analyst','admin','viewer'):
            r=cls.client.post('/auth/login',json={'email':role+'@shuangleng.demo','password':'Shuangleng2026!'})
            assert r.status_code==200,r.text;cls.tokens[role]={'Authorization':'Bearer '+r.json()['token']}
    def test_01_studies_expose_mode_support(self):
        r=self.client.get('/studies',headers=self.tokens['analyst']);self.assertEqual(r.status_code,200)
        by={s['code']:s for s in r.json()}
        self.assertTrue(by['SL-04']['multi_supported']);self.assertFalse(by['SL-02']['multi_supported'])
        self.assertIn('小群',by['SL-02']['multi_reason'])
        self.assertFalse(by['SL-03']['supported'])
    def test_02_default_run_mode_is_single_for_compatibility(self):
        study=[s for s in self.client.get('/studies',headers=self.tokens['analyst']).json() if s['code']=='SL-01'][0]
        r=self.client.post('/runs',headers=self.tokens['analyst'],json={'name':'默认模式','study_id':study['id'],'alpha':.05,'seed':2026})
        self.assertEqual(r.status_code,202,r.text);self.assertEqual(r.json()['mode'],'single')
    def test_03_multi_run_export_requires_approval_and_carries_attribution(self):
        study=[s for s in self.client.get('/studies',headers=self.tokens['analyst']).json() if s['code']=='SL-04'][0]
        r=self.client.post('/runs',headers=self.tokens['analyst'],json={'name':'多群导出','study_id':study['id'],'alpha':.05,'seed':2026,'mode':'multi'})
        self.assertEqual(r.status_code,202,r.text);rid=r.json()['id'];self.assertEqual(r.json()['mode'],'multi')
        with database.SessionLocal() as db:
            row=db.get(DeviationRun,rid);row.result=analyze_deviation(row.input_snapshot,.05,2026);row.status='completed';db.commit()
        self.assertEqual(self.client.get(f'/runs/{rid}/export',headers=self.tokens['viewer']).status_code,403)
        note=self.client.post('/notes',headers=self.tokens['analyst'],json={'run_id':rid,'title':'多群审阅','interpretation':'群归属与同群校准的冻结说明。','proposal':'按群分别独立补测并记录归属间隔。'}).json()
        self.client.patch(f"/notes/{note['id']}/review",headers=self.tokens['admin'],json={'status':'approved','review_note':'批准多群归档，保持冻结归属与逐群校准。'})
        export=self.client.get(f'/runs/{rid}/export',headers=self.tokens['viewer'])
        self.assertEqual(export.status_code,200)
        header=export.text.lstrip('﻿').splitlines()[0]
        for column in ('group','group_cal_m','group_assignment_basis','assignment_margin','ambiguous','mode'):
            self.assertIn(column,header)
        self.assertTrue(self.client.get(f'/runs/{rid}',headers=self.tokens['analyst']).json()['result']['group_count']>=1)
    def test_04_degenerate_multi_is_rejected_at_creation(self):
        study=[s for s in self.client.get('/studies',headers=self.tokens['analyst']).json() if s['code']=='SL-03'][0]
        r=self.client.post('/runs',headers=self.tokens['analyst'],json={'name':'退化拒绝','study_id':study['id'],'alpha':.05,'seed':2026,'mode':'multi'})
        self.assertEqual(r.status_code,422,r.text)
        self.assertNotIn('NaN',r.text);self.assertNotIn('Infinity',r.text)
    def test_05_viewer_cannot_create_multi_run(self):
        study=[s for s in self.client.get('/studies',headers=self.tokens['viewer']).json() if s['code']=='SL-04'][0]
        r=self.client.post('/runs',headers=self.tokens['viewer'],json={'name':'越权创建','study_id':study['id'],'alpha':.05,'seed':2026,'mode':'multi'})
        self.assertEqual(r.status_code,403)
    def test_06_invalid_mode_rejected_without_echoing_input(self):
        r=self.client.post('/runs',headers=self.tokens['analyst'],json={'name':'非法模式','study_id':1,'alpha':.05,'seed':2026,'mode':'tri'})
        self.assertEqual(r.status_code,422);self.assertNotIn('tri',r.text)

class UpgradeTest(unittest.TestCase):
    def test_historical_database_gains_mode_single(self):
        path='/tmp/devtest-history.db'
        if os.path.exists(path):os.remove(path)
        engine=create_engine('sqlite:///'+path)
        with engine.begin() as c:
            c.execute(text("CREATE TABLE deviation_runs (id INTEGER PRIMARY KEY, name VARCHAR, study_id INTEGER, alpha FLOAT, seed INTEGER, input_snapshot JSON, status VARCHAR, result JSON, error TEXT, owner VARCHAR, created_at VARCHAR)"))
            c.execute(text("INSERT INTO deviation_runs (name,study_id,alpha,seed,status,result,owner) VALUES ('旧单峰',1,0.05,2026,'completed','{}','温砚')"))
        import src.config.database as dbmod
        original=dbmod.engine;dbmod.engine=engine
        try:dbmod.upgrade_schema()
        finally:dbmod.engine=original
        with engine.begin() as c:
            self.assertEqual(c.execute(text('SELECT mode FROM deviation_runs')).scalar_one(),'single')
            dbmod.engine=engine;dbmod.upgrade_schema();dbmod.engine=original

if __name__=='__main__':unittest.main()
