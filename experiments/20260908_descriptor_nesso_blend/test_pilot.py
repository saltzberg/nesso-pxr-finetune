import unittest
import numpy as np
import run as P
from verify import lp_alpha

class BlendTests(unittest.TestCase):
    def test_endpoints_and_identical(self):
        y=np.array([1.,2.,3.]); w=np.ones(3)
        self.assertEqual(P.blend_weight(y,y,y+1,w)[0],0)
        self.assertEqual(P.blend_weight(y,y+1,y,w)[0],1)
        self.assertEqual(P.blend_weight(y,y,y,w)[0],0)
    def test_interior_and_lp(self):
        rng=np.random.default_rng(42)
        for _ in range(8):
            y,d,r=rng.normal(size=(3,20)); w=np.exp(rng.normal(size=20))
            a,_,loss=P.blend_weight(y,d,r,w); b,objective=lp_alpha(y,d,r,w)
            self.assertAlmostEqual(a,b,places=10)
            self.assertAlmostEqual(loss.min(),objective,places=12)
    def test_invalid(self):
        with self.assertRaises(ValueError): P.blend_weight([1],[1],[np.nan],[1])
        with self.assertRaises(ValueError): P.blend_weight([1],[1],[1],[0])
    def test_group_nesting(self):
        groups=np.repeat(np.arange(30).astype(str),3)
        folds,_=P._folds({'groups':groups},len(groups),3,42)
        seen=[]
        for tr,va in folds:
            self.assertFalse(set(groups[tr])&set(groups[va])); seen.extend(va)
            inner,_=P._folds({'groups':groups[tr]},len(tr),3,42)
            for a,b in inner:
                self.assertFalse(set(groups[tr[a]])&set(groups[tr[b]]))
                self.assertFalse(set(tr[a])&set(va))
        np.testing.assert_array_equal(sorted(seen),np.arange(len(groups)))
    def test_stable_preprocessing_training_only(self):
        rng=np.random.default_rng(42); x=rng.normal(size=(20,768)); w=np.ones(20)
        p=P.R.fit_preprocessing(x,w,True)
        np.testing.assert_allclose(p['mean'],x.mean(axis=0),atol=1e-15)
        self.assertTrue(np.all(p['scale']>=.1*p['global_spread']))
        before=p['mean'].copy(); P.R.transform(np.full((5,768),1e9),p)
        np.testing.assert_array_equal(before,p['mean'])

if __name__=='__main__': unittest.main()
