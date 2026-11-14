import os,sys,unittest,csv,io
os.environ.setdefault('DATABASE_URL','sqlite:////tmp/devtest-render.db')
os.environ.setdefault('REDIS_URL','redis://localhost:6399/0')
sys.path[:0]=['/workspace/backend','/workspace']
from sqlalchemy import create_engine;from sqlalchemy.orm import sessionmaker
import src.config.database as database
if os.path.exists('/tmp/devtest-render.db'):os.remove('/tmp/devtest-render.db')
engine=create_engine(os.environ['DATABASE_URL'],connect_args={'check_same_thread':False});database.engine=engine;database.SessionLocal=sessionmaker(bind=engine,expire_on_commit=False)
from src.models import ArchiveStudy,DeviationRun,ReviewNote
from src.services.study_service import create_snapshot
from src.services.analysis_service import analyze_deviation
from src.services.run_service import export_run
from scripts.seed_demo import seed

class ResultShapeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed()
        with database.SessionLocal() as db:
            cls.results={};cls.rows={}
            for code,mode in (('SL-04','multi'),('SL-01','single')):
                study=db.query(ArchiveStudy).filter_by(code=code).first()
                result=analyze_deviation(create_snapshot(db,study,mode),.05,2026)
                cls.results[mode]=result
                run=DeviationRun(name='形状检查 '+mode,study_id=study.id,alpha=.05,seed=2026,mode=mode,input_snapshot={'config':{'mode':mode}},status='completed',result=result,owner='测试')
                db.add(run);db.flush();db.add(ReviewNote(run_id=run.id,title='形状检查批准',interpretation='测试导出形状。',proposal='核对列顺序与归属字段。',status='approved',owner='测试',reviewer='测试',review_note='批准形状检查。'));db.flush()
                cls.rows[mode]=run.id
            db.commit()
    def test_multi_has_no_single_fit_and_each_group_has_full_model(self):
        r=self.results['multi']
        self.assertNotIn('fit',r)
        for group in r['groups']:
            for key in ('location_scaled','covariance_scaled','raw_support_mask','support_mask','train_distances','train_windows','calibration_ecdf'):
                self.assertIn(key,group)
            self.assertEqual(len(group['train_windows']),group['train_count'])
    def test_multi_ecdf_is_per_group_and_matches_group_m(self):
        r=self.results['multi']
        for group in r['groups']:
            self.assertEqual(sum(1 for c in r['calibration'] if c['group_code']==group['group_code']),group['cal_count'])
            self.assertAlmostEqual(group['calibration_ecdf'][-1]['cdf'],1.0)
            self.assertAlmostEqual(group['resolution'],1/(group['cal_count']+1))
    def test_single_result_keeps_legacy_shape(self):
        r=self.results['single']
        self.assertIn('fit',r);self.assertNotIn('groups',r)
        self.assertGreater(len(r['calibration_ecdf']),0)
        self.assertEqual(r['selected_count'],sum(x['selected'] for x in r['test']))
    def test_single_export_keeps_legacy_columns_in_order(self):
        with database.SessionLocal() as db:
            rows=list(csv.DictReader(io.StringIO(export_run(db,self.rows['single']).lstrip('﻿'))))
        self.assertEqual(list(rows[0].keys()),['window','split','temperature','humidity','particles','variation','D_squared','cal_ge_count','upper_tail_p','preset_alpha','selected','cal_m','seed','mode','fit_hash','input_hash'])
        self.assertEqual(rows[0]['mode'],'single')
    def test_multi_export_appends_attribution_columns(self):
        with database.SessionLocal() as db:
            content=export_run(db,self.rows['multi']).lstrip('﻿');rows=list(csv.DictReader(io.StringIO(content)))
        head=list(rows[0].keys())
        self.assertEqual(head[:9],['window','split','temperature','humidity','particles','variation','D_squared','cal_ge_count','upper_tail_p'])
        for column in ('group','group_cal_m','group_assignment_basis','second_group','second_group_D_squared','assignment_margin','ambiguous','mode'):
            self.assertIn(column,head)
        start=head.index('group');self.assertEqual(head[start:start+7],['group','group_cal_m','group_assignment_basis','second_group','second_group_D_squared','assignment_margin','ambiguous'])
        self.assertEqual(head[-5:],['cal_m','seed','mode','fit_hash','input_hash'])
        self.assertTrue(all(row['group'].startswith('G') for row in rows))
        self.assertTrue(all(row['mode']=='multi' for row in rows))
        self.assertEqual(len(rows),self.results['multi']['counts']['test'])

if __name__=='__main__':unittest.main()
