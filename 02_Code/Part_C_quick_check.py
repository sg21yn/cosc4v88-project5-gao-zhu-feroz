import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np


def check(out: Path):
    files = ['client_flags.csv', 'flag_evaluation.csv', 'ablation.csv',
             'threshold_sensitivity.csv', 'baselines.csv', 'demo_cases.csv',
             'method_config.json']
    for f in files:
        assert (out / f).exists(), f'Missing {f}'
    clients = pd.read_csv(out / 'client_flags.csv')
    ev = pd.read_csv(out / 'flag_evaluation.csv')
    abl = pd.read_csv(out / 'ablation.csv')
    sens = pd.read_csv(out / 'threshold_sensitivity.csv')
    bases = pd.read_csv(out / 'baselines.csv')
    demos = pd.read_csv(out / 'demo_cases.csv')
    config = json.loads((out / 'method_config.json').read_text())
    models = ['LogisticRegression', 'XGBoost']
    assert len(clients) == 200 and clients.ID.nunique() == 100
    assert clients.groupby('ID').split.nunique().eq(1).all(), 'Same ID assigned to different splits'
    
    for model in models:
        m = clients[clients.model == model]
        assert len(m) == 100 and m.ID.nunique() == 100
        assert m.split.str.startswith('calibration').sum() == 50
        assert m.split.str.startswith('evaluation').sum() == 50
        assert np.isfinite(m.score).all() and m.score.between(0,1).all()
        cutoff = config['calibration'][model]['cutoff']
        assert (m.flag == (m.score >= cutoff).astype(int)).all(), 'Flags differ from recorded cutoff'
        assert (m.bad_deletion == (m.lime_deletion_top3 <= m.random_deletion_3).astype(int)).all()
        assert len(abl[abl.model == model]) == 5, 'Expected all-four + four leave-one-out ablations'
        assert len(sens[sens.model == model]) == 4, 'Expected four threshold settings'
        assert len(bases[bases.model == model]) == 3, 'Expected three single-signal baselines'

    for model, expected in {'LogisticRegression':(13, 11, 9, 2),
                            'XGBoost':(15, 7, 5, 2)}.items():
        r=ev[(ev.model==model)&(ev.partition=='evaluation')].iloc[0]
        actual=tuple(int(r[k]) for k in ['flagged','bad','caught','missed'])
        assert actual==expected, f'{model} differs from saved run: {actual} != {expected}'
        print(f'PASS {model}: 50 evaluation clients; reviews, bad, caught, missed = {actual}')
    
    def demo(id_,model):
        rows=demos[(demos.ID==id_)&(demos.model==model)]
        
        assert len(rows)==1, f'Demo client {id_} ({model}) missing'
        return rows.iloc[0]
    assert int(demo(2072,'XGBoost').flag)==0 and demo(2072,'XGBoost')['split'].startswith('evaluation')
    assert int(demo(10157,'XGBoost').flag)==1 and demo(10157,'XGBoost')['split'].startswith('evaluation')
    assert int(demo(20345,'LogisticRegression').flag)==1 and demo(20345,'LogisticRegression')['split'].startswith('calibration')
    assert int(config['first_pass_client_XGBoost'])==2072
    
    print('PASS 100-ID join / client-level 50-50 split / score range / cutoff rules')
    print('PASS 4 threshold settings, 4 ablations, 3 baselines per model')
    print('PASS demo provenance: 2072 held-out pass; 10157 held-out warning; 20345 calibration warning')
    print('NOTE: Both halves originate from the original test explanations, not val.csv.')
    print('NOTE: Ablation evaluation review counts need not match across methods.')

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    check(args.output_dir)
