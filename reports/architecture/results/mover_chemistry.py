from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from rdkit import Chem,rdBase
from rdkit.Chem import Descriptors,rdMolDescriptors,Crippen,Draw
H=Path(__file__).resolve().parent
allrows=pd.read_csv(H/'proximity_all.csv');top=pd.read_csv(H/'proximity_top20.csv');d=allrows.drop_duplicates('record_id').copy()
patterns={'nonamide_amine':'[NX3;!$(N-C=O);!$(N-S(=O)=O);!$(N-P=O);!$(N=*)]','carboxylic_acid':'[CX3](=O)[OX2H1,OX1-]'}
patterns={k:Chem.MolFromSmarts(v) for k,v in patterns.items()}
rows=[];mols={}
for _,r in d.iterrows():
 m=Chem.MolFromSmiles(r.smiles);assert m is not None;mols[r.record_id]=m
 rows.append(dict(record_id=r.record_id,MW=Descriptors.MolWt(m),heavy_atoms=m.GetNumHeavyAtoms(),cLogP=Crippen.MolLogP(m),TPSA=rdMolDescriptors.CalcTPSA(m),rings=rdMolDescriptors.CalcNumRings(m),rotatable=rdMolDescriptors.CalcNumRotatableBonds(m),formal_charge=Chem.GetFormalCharge(m),**{k:m.HasSubstructMatch(p) for k,p in patterns.items()}))
d=d.merge(pd.DataFrame(rows),on='record_id',validate='one_to_one');d['MW_lt200']=d.MW<200;d['pEC50_lt4']=d.y_true<4
metrics=['MW','heavy_atoms','cLogP','TPSA','rings','rotatable','y_true'];flags=['MW_lt200','nonamide_amine','carboxylic_acid','pEC50_lt4'];out=[]
def summarise(x,label):
 out.append(dict(selection=label,n=len(x),**{k:float(x[k].median()) for k in metrics},**{k:int(x[k].sum()) for k in flags}))
summarise(d,'all');summarise(d[d.y_true<4],'all_pEC50_lt4')
for (n,c),x in top.groupby(['budget','comparison']):
 a=d[d.record_id.isin(x.record_id)];assert len(a)==20;summarise(a,f'N{n}_{c}')
u=top[top.comparison.isin(['head_only-minus-native','head_plus_lora-minus-native'])].record_id.unique();summarise(d[d.record_id.isin(u)],'native_mover_unique_union')
d.to_csv(H/'mover_chemistry_all.csv',index=False);pd.DataFrame(out).to_csv(H/'mover_chemistry_summary.csv',index=False)
# Canonical/source and prepared state are distinct; inspect exact original metadata.
manifest=pd.read_csv(H.parents[2]/'data/published/modeling_manifest.csv');sel=manifest[manifest.record_id.isin(top.record_id)].copy();sel.to_csv(H/'mover_chemistry_source_metadata.csv',index=False)
print(pd.DataFrame(out).to_string(index=False))
print('\nEXEMPLAR METADATA');print(sel[sel.record_id.isin(['nesso_4f5d960ee79593ed1c2e','nesso_56ced9f6b257bd2a4a19','nesso_99543dc096e020200214','nesso_9488ed207b0172efb5ab','nesso_97fd17ddc3adcd3d769a'])][['record_id','original_id','raw_smiles','prepared_smiles','formal_charge','pEC50_standard_error']].to_string(index=False))
view=top[(top.budget==500)&(top.comparison=='head_plus_lora-minus-native')].head(12)
img=Draw.MolsToGridImage([mols[i] for i in view.record_id],molsPerRow=4,subImgSize=(320,240),legends=[f'{i}\npEC50 {y:.2f}; MW {d.set_index("record_id").loc[i,"MW"]:.0f}' for i,y in zip(view.record_id,view.y_true)])
img.save(str(H/'mover_chemistry_examples.png'))
(H/'mover_chemistry_verification.json').write_text(json.dumps({'status':'PASS','unique_compounds':len(d),'native_top20_union':len(u),'all_six_top20_union':int(top.record_id.nunique()),'rdkit':rdBase.rdkitVersion,'scope':'retrospective descriptive chemistry; no fits; motifs are substructure matches, not pKa/charge or mechanism assignments','inputs':{n:hashlib.sha256((H/n).read_bytes()).hexdigest() for n in ['proximity_all.csv','proximity_top20.csv']}},indent=2))
