#!/usr/bin/env python3
"""Post-fit descriptive addendum; no training, selection, or protocol amendment."""
import run as P
import numpy as np

def main():
    O=P.OUT; f=P.C.csv(O/'test_predictions.csv'); result={}; total=f.weight.sum()
    for name,mask in [('lt4',f.y_true<4),('ge4',f.y_true>=4)]:
        g=f[mask]; diff=abs(g.blend-g.y_true)-abs(g.descriptor-g.y_true)
        result[name]={'n':len(g),'weight_share':g.weight.sum()/total,'weighted_delta':np.average(diff,weights=g.weight),'raw_delta':diff.mean(),'primary_contribution':np.sum(diff*g.weight)/total}
    delta=P.C.read_json(O/'metrics.json')['blend']['weighted_mae']-P.C.read_json(O/'metrics.json')['descriptor']['weighted_mae']
    np.testing.assert_allclose(sum(v['primary_contribution'] for v in result.values()),delta,atol=1e-12,rtol=0)
    P.put(O/'descriptive_stratum_decomposition.json',result)
    before={str(p):P.sha(p) for p in O.rglob('*') if p.is_file()}
    try: P.run()
    except ValueError as e: assert 'refit' in str(e)
    else: raise AssertionError('repeat launch did not refuse fitting')
    assert before=={str(p):P.sha(p) for p in O.rglob('*') if p.is_file()}
    P.put(O/'refit_guard_verification.json',{'status':'PASS','repeat_run_refused':True,'all_existing_outputs_byte_identical':True,'addendum_code_sha256':P.sha(__file__)})
    print('descriptive decomposition reconciled; repeated run refused without changing any artifact')

if __name__=='__main__': main()
