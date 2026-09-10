"""Independent summary checks + real byte replay; no predictor refits."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,hashlib,subprocess,tempfile,sys,unittest
import numpy as np
import pandas as pd
from scipy.stats import rankdata
E=Path(__file__).resolve().parent;R=E.parents[1];A=E/'artifacts'
def read(n):return pd.read_csv(A/n,float_precision='round_trip')
class Checks(unittest.TestCase):
 def test_sources(self):
  for n,h in json.loads((A/'source_hashes.json').read_text()).items():self.assertEqual(hashlib.sha256((R/n).read_bytes()).hexdigest(),h)
 def test_identity_and_values(self):
  from rdkit import Chem
  from rdkit.Chem import Descriptors
  d=read('merged_compounds.csv'); self.assertTrue(np.array_equal(d.MW,d.identity_canonical_smiles.map(lambda s:Descriptors.MolWt(Chem.MolFromSmiles(s)))))
  self.assertEqual(len(d),3344);self.assertTrue(d.record_id.is_unique);self.assertEqual(int(d.small.sum()),237);self.assertEqual(int(d.emax_qualified.sum()),3072);self.assertTrue((d.raw_pEC50_SE==d.pEC50_standard_error).all());self.assertTrue((d.Emax_ratio.ge(.6)==d.emax_qualified).all())
 def test_independent_statistics(self):
  d=read('merged_compounds.csv');s=read('stratum_summaries.csv');c=read('primary_contrasts.csv');audit=json.loads((A/'audit.json').read_text());lo,hi=audit['common_support']
  for row in c.itertuples():
   z=d if row.scope=='full' else d[d.native.between(lo,hi)];a=sorted(z.loc[z.small,row.outcome]);b=sorted(z.loc[~z.small,row.outcome]);self.assertEqual((len(a),len(b)),(row.n_small,row.n_large));self.assertAlmostEqual(float(np.median(a)-np.median(b)),row.delta_median,places=14)
  for row in read('associations.csv').itertuples():
   z={'full':d,'small':d[d.small],'large':d[~d.small],'common_support':d[d.native.between(lo,hi)]}[row.scope];self.assertAlmostEqual(float(z[[row.x,row.y]].rank().corr().iloc[0,1]),row.rho,places=12)
  for row in read('adjusted_and_correction_associations.csv').fillna('').itertuples():
   controls=row.controls.split(',') if row.controls else [];X=np.column_stack([np.ones(len(d))]+[rankdata(d[x]) for x in controls]);Q=np.linalg.qr(X)[0];x=rankdata(d[row.x]);y=rankdata(d[row.y]);self.assertAlmostEqual(float(np.corrcoef(x-Q@(Q.T@x),y-Q@(Q.T@y))[0,1]),row.rho,places=12)
 def test_byte_replay(self):
  with tempfile.TemporaryDirectory(prefix='replay_',dir=E) as td:
   subprocess.run([sys.executable,str(E/'run.py'),'--out',td],check=True,stdout=subprocess.DEVNULL)
   for p in A.iterdir():self.assertEqual(p.read_bytes(),(Path(td)/p.name).read_bytes(),p.name)
if __name__=='__main__':
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks)); receipt={'status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,'independent_checks':'source integrity, identities/counts/qualification/SE, all rank associations, QR partial ranks, primary medians/counts','byte_replay':'all files in artifacts, including PNGs; no inference or predictor fitting'};(E/'verification.json').write_text(json.dumps(receipt,indent=2));sys.exit(not result.wasSuccessful())
