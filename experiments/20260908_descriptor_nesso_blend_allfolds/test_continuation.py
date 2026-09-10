import sys
sys.dont_write_bytecode=True
import importlib, unittest, pickle
A=importlib.import_module('experiments.20260908_descriptor_nesso_blend_allfolds.run')
sys.modules['verify']=A.V
spec=importlib.util.spec_from_file_location('pilot_tests',A.PILOT_EXP/'test_pilot.py')
T=importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
BlendTests=T.BlendTests

class ContinuationTests(unittest.TestCase):
    def test_same_functions(self):
        self.assertEqual(A.P.fit_one.__module__,'experiments.20260908_descriptor_nesso_blend.run')
        self.assertEqual(A.P.blend_weight.__module__,'experiments.20260908_descriptor_nesso_blend.run')
    def test_roles_and_selector(self):
        _,assign,cells=A.P.C.strict_identity()
        selected=[c for c in cells if c['n_train']==500 and c['draw']==0]
        self.assertEqual(sorted(c['outer_fold'] for c in selected),list(range(5)))
        test=[]
        for c in selected:
            f,cal,t=(c['roles'][r] for r in ('fit','calibration','test'))
            self.assertEqual((len(f),len(cal)),(400,100))
            self.assertFalse(set(f)&set(cal))
            self.assertFalse((set(f)|set(cal))&set(t)); test.extend(t)
        self.assertEqual(len(test),3344); self.assertEqual(len(set(test)),3344)
    def test_real_serialization_and_no_refit(self):
        A.configure(0)
        before=A.inventory(A.PILOT)
        old=A.P.fit_one
        def forbidden(*a,**k): raise AssertionError('FIT CALLED')
        A.P.fit_one=forbidden
        try:
            with self.assertRaisesRegex(ValueError,'Output already exists'): A.P.run()
        finally: A.P.fit_one=old
        for kind in ('descriptor','ridge'):
            path=A.PILOT/'models'/('final_'+kind)/'model.pkl'
            state=pickle.loads(path.read_bytes())
            self.assertIn('method',pickle.loads(pickle.dumps(state)))
        self.assertEqual(before,A.inventory(A.PILOT))

if __name__=='__main__': unittest.main()
