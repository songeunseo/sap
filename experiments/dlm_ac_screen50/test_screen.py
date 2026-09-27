import itertools
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from experiments.dlm_multiscale_ac50.artifacts import read,freeze,write,digest,ReadoutStore
from experiments.dlm_multiscale_ac50.core import make_bank,metrics,paired,holm
from experiments.dlm_multiscale_ac50.evaluation import evaluate_requests,read_predictions
from .core import ARMS,square_bank,square_metrics,vector_pair,mean_rows,proposals,accept
from .prepare import diagnostic_pairs,memory_estimate,MULTI
from .run import job_graph,select_ready,terminate_owned,proc_identity

class TheoryTests(unittest.TestCase):
    def test_square_modes_and_telescoping(self):
        for e in [[1,1,1,1],[1,-1,-1,1],[0,1,2,4]]:
            x=np.array(e,dtype=float).reshape(1,4,1);m=square_metrics(x,np.zeros_like(x))['mean']
            self.assertAlmostEqual(m['A'],m['common2']+m['main1_2']+m['main2_2']+m['interaction2'])
            self.assertAlmostEqual(m['C'],2*m['main1_2']+2*m['main2_2']+4*m['interaction2'])
            self.assertEqual((e[1]-e[0])+(e[3]-e[1]),(e[2]-e[0])+(e[3]-e[2]))
        m=square_metrics(np.array([1,-1,-1,1]).reshape(1,4,1),np.zeros((1,4,1)))['mean']
        self.assertEqual((m['A'],m['C'],m['AC'],m['mixed2']),(1,4,5,16))
        self.assertEqual(square_metrics(np.ones((40,4,8)),np.ones((40,4,8)))['mean']['AC'],0)
    def test_square_seed_and_degenerate(self):
        source=dict(mask_id=999,states=[dict(sequence_index=i,clean_ids=[list(range(256))]) for i in range(8)])
        a=square_bank(source,'calibration');self.assertEqual(a,square_bank(source,'calibration'));self.assertEqual(a['states'],160)
        self.assertNotEqual(a,square_bank(source,'diagnostic'))
        from .core import validate_square
        q=a['quartets'][0];eligible=q['eligible'];q['p']=1.;q['groups']=[[],[]];q['group_size']=0
        for n in q['nodes']:
            n['visible']=eligible;n['masked_count']=8;n['input_ids']=[i if i in eligible else 999 for i in range(256)]
        validate_square(a)
    def test_vector_shift_chunk_margin_identity(self):
        rng=np.random.default_rng(8);p=rng.normal(size=(2,3,17)).astype('float32');d=rng.normal(size=(2,3,17)).astype('float32')
        a=vector_pair(p,d,17);b=vector_pair(p,d,4)
        np.testing.assert_allclose(list(a.values()),list(b.values()),rtol=1e-9,atol=1e-10)
        shifted=vector_pair(p.astype(float)+np.array([7.,-13.])[:,None,None],d.astype(float)+np.array([-3.,8.])[:,None,None],5)
        np.testing.assert_allclose(list(a.values()),list(shifted.values()),rtol=1e-9,atol=1e-10)
        e=p[0].astype(float)-d[0];v=17;left=np.mean((e-e.mean(1)[:,None])**2);right=np.mean([sum((r[i]-r[j])**2 for i in range(v) for j in range(i+1,v))/v**2 for r in e]);self.assertAlmostEqual(left,right)
    def test_vector_wrong_center_and_query_weights(self):
        p=np.array([[[1.,1.,3.,3.]],[[1.,1.,3.,3.]]]);d=np.zeros_like(p)
        self.assertEqual(vector_pair(p,d,2)['A'],1)
        bad=p.copy();bad[:,:,:2]-=bad[:,:,:2].mean(2,keepdims=True);bad[:,:,2:]-=bad[:,:,2:].mean(2,keepdims=True)
        self.assertEqual(float(np.mean(bad**2)),0)
        rows=[dict(A=1.,C=0.,AC=1.),dict(A=9.,C=0.,AC=9.)];self.assertEqual(mean_rows(rows)['A'],5.)
    def test_wrong_token_redistribution(self):
        dense=np.log(np.array([[[.4,.5,.1]],[[.4,.1,.5]]]))
        pred=np.repeat(dense[:1],2,axis=0)
        self.assertGreater(vector_pair(pred,dense)['C'],0)
        self.assertEqual(dense[0,0,0],dense[1,0,0])
    def test_multi_frozen_and_bounds(self):
        bank=read(MULTI/'bank_calibration.json');e=np.tile([1,1,-1,-1,1,1,-1,-1],16)[:,None]*np.ones((1,8));m=metrics(e,np.zeros_like(e),bank)['mean']
        self.assertAlmostEqual(m['Multi'],7/3);self.assertEqual(m['C1'],0);self.assertEqual(m['C2'],4);self.assertEqual(m['C4'],0)
        rng=np.random.default_rng(9);e=rng.normal(size=(128,8));m=metrics(e,np.zeros_like(e),bank)['mean'];var=e.reshape(16,8,8).var(axis=1).mean();cm=m['Multi']-m['A']
        self.assertGreaterEqual(cm,4*var/3);self.assertLessEqual(cm,4*var);self.assertAlmostEqual(m['C_all'],16*var/7)
        c=read(MULTI/'config.json');source=read(c['plan']['sources']['calibration_source']['path'])
        self.assertEqual(make_bank(source,c['plan']['bank'],'calibration'),bank)
    def test_cross_bias_finite_enumeration(self):
        p,r=.2,.7;rho=math.sqrt(p*(1-r)/(r*(1-p)));f=lambda x:1+2*x[0]-x[1]+3*x[0]*x[1]
        coeff=[]
        for t in [p,r]:
            cc=[]
            for subset in [(),(0,),(1,),(0,1)]:
                cc.append(sum(math.prod(t if b else 1-t for b in x)*f(x)*math.prod((x[i]-t)/math.sqrt(t*(1-t)) for i in subset) for x in itertools.product([0,1],repeat=2)))
            coeff.append(cc)
        lhs=0
        for states in itertools.product([(0,0,1-r),(0,1,r-p),(1,1,p)],repeat=2):
            x=[s[0] for s in states];y=[s[1] for s in states];lhs+=math.prod(s[2] for s in states)*(f(y)-f(x))**2
        rhs=sum(a*a+b*b-2*rho**k*a*b for a,b,k in zip(*coeff,[0,1,1,2]));self.assertAlmostEqual(lhs,rhs)
    def test_diagnostic_recipe(self):
        c=read(MULTI/'config.json');cal=read(c['plan']['sources']['calibration_source']['path']);diag=read(c['plan']['sources']['diagnostic_source']['path'])
        x=diagnostic_pairs(cal,diag);self.assertEqual(len(x['pairs']),80);self.assertEqual(x,diagnostic_pairs(cal,diag))
        for p in x['pairs']:
            self.assertFalse(set(p['query'])&set(p['reveal']))
            self.assertTrue(all(p['before'][0][i]==diag['mask_id']==p['after'][0][i] for i in p['query']))
        m=memory_estimate({'calibration':x,'diagnostic':x});self.assertEqual(m['total_teacher_bytes'],2*m['teacher_bytes']['calibration'])
    def test_exchange_budget_bounds_sign_and_acceptance(self):
        cfg=read(MULTI/'config.json');refs=read(cfg['legacy_manifests']['AC']['path'])['entries'];anchor=[r['selected_mask']['prune_per_row'] for r in refs];costs=list(range(32))
        ps=proposals(anchor,anchor,refs,costs);self.assertTrue(ps)
        total=sum(k*r['shape'][0] for k,r in zip(anchor,refs))
        for p in ps:
            self.assertEqual(total,sum(k*r['shape'][0] for k,r in zip(p['row_counts'],refs)))
            self.assertEqual(p['predicted_gain'],53248*41*(costs[p['receiver']]-costs[p['donor']]))
            changed={i//7 for i,(x,y) in enumerate(zip(anchor,p['row_counts'])) if x!=y};self.assertEqual(changed,{p['donor'],p['receiver']})
        self.assertIsNone(accept(1,[1,1.1],1e-5));self.assertEqual(accept(1,[.9,.8],1e-5),1);self.assertIsNone(accept(1,[],1e-5))
        with self.assertRaises(ValueError):accept(1,[float('nan')],1e-5)

class ExecutionTests(unittest.TestCase):
    def test_scalar_checkpoint_fingerprint(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.json';s=ReadoutStore(p,{'a':1},2,2);s.append([1,2]);s=ReadoutStore(p,{'a':1},2,2);s.append([3,4]);self.assertEqual(s.complete(),[[1.,2.],[3.,4.]])
            with self.assertRaises(RuntimeError):ReadoutStore(p,{'a':2},2,2)
    def test_document_kill_resume(self):
        with tempfile.TemporaryDirectory() as d:
            req=[dict(example_id=i,doc_hash=str(i),prompt_hash=str(i),target_hash=str(i),reference_answer='1') for i in range(4)]
            calls=[]
            def gen(q):
                calls.append(q['example_id'])
                if len(calls)==3:raise RuntimeError('simulated death')
                return '1'
            grade=lambda q,t:dict(extracted_answer='1',correct=True)
            with self.assertRaises(RuntimeError):evaluate_requests(d,req,{'physical':'x'},'p','m',gen,grade,lambda *a,**kw:None)
            hashes={p.name:p.read_bytes() for p in (Path(d)/'examples').glob('*.json')};again=[]
            evaluate_requests(d,req,{'physical':'x'},'p','m',lambda q:again.append(q['example_id']) or '1',grade,lambda *a,**kw:None)
            self.assertEqual(again,[2,3]);self.assertTrue(all((Path(d)/'examples'/n).read_bytes()==b for n,b in hashes.items()))
            with self.assertRaises(RuntimeError):read_predictions(d,req,{'physical':'wrong'},'p')
    def test_scheduler_one_two_four_and_resume(self):
        jobs=job_graph()
        for slots in [1,2,4]:
            done=set();active=set();last=None;seen=[]
            while len(done)<len(jobs):
                for _ in range(slots):
                    j=select_ready(jobs,done,active,last)
                    if j is None:break
                    self.assertTrue(set(j['deps'])<=done);self.assertNotIn(j['id'],active);active.add(j['id']);seen.append(j['id']);last=j['family']
                self.assertTrue(active);done.update(active);active.clear()
                # Restart persistence: completed set survives a round-trip.
                done=set(json.loads(json.dumps(sorted(done))))
            self.assertEqual(len(seen),len(set(seen)));self.assertTrue(set(ARMS)<=done)
    def test_stop_leaves_unrelated_process(self):
        a=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
        b=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
        try:
            terminate_owned([dict(pid=a.pid,start_ticks=proc_identity(a.pid))]);a.wait(timeout=3);self.assertIsNone(b.poll())
        finally:
            if a.poll() is None:a.kill()
            b.kill();b.wait();a.wait()
    def test_paired_and_holm(self):
        x=paired([False,True,True,False],[True,False,True,True]);self.assertEqual((x['gained'],x['lost']),(2,1))
        v={str(i):dict(exact_mcnemar_p=.01*(i+1)) for i in range(7)};holm(v);self.assertEqual(v['0']['holm_p'],.07)
    def test_help_no_torch_or_gpu(self):
        code="import sys;from unittest.mock import patch;from experiments.dlm_ac_screen50.run import main;sys.argv=['run','--help'];\nwith patch('subprocess.check_output',side_effect=AssertionError('GPU polling')):\n try:main()\n except SystemExit:pass\nassert 'torch' not in sys.modules"
        subprocess.run([sys.executable,'-B','-c',code],check=True,stdout=subprocess.DEVNULL)

if __name__=='__main__':unittest.main()
