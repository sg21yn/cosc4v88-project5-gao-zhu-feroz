import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

# v2: checks the re-run on the 500-client sample (200 val calibration / 300 test evaluation)

def check(out: Path):
    files = ['client_flags.csv', 'flag_evaluation.csv', 'ablation.csv',
             'threshold_sensitivity.csv', 'baselines.csv', 'demo_cases.csv',
             'method_config.json', 'workload_matched.csv']
    for f in files:
        assert (out / f).exists(), f'Missing {f}'
    clients = pd.read_csv(out / 'client_flags.csv')
    ev = pd.read_csv(out / 'flag_evaluation.csv')
    abl = pd.read_csv(out / 'ablation.csv')
    sens = pd.read_csv(out / 'threshold_sensitivity.csv')
    bases = pd.read_csv(out / 'baselines.csv')
    config = json.loads((out / 'method_config.json').read_text())
    assert len(clients) == 1000 and clients.ID.nunique() == 500
    assert clients.groupby('ID').split.nunique().eq(1).all(), 'Same ID assigned to different splits'
    for model in ['LogisticRegression', 'XGBoost']:
        m = clients[clients.model == model]
        assert len(m) == 500
        assert (m.split.str.startswith('calibration') == (m.data_split == 'val')).all()
        assert m.split.str.startswith('calibration').sum() == 200
        assert m.split.str.startswith('evaluation').sum() == 300
        assert np.isfinite(m.score).all() and m.score.between(0, 1).all()
        cutoff = config['calibration'][model]['cutoff']
        assert (m.flag == (m.score >= cutoff).astype(int)).all(), 'Flags differ from recorded cutoff'
        assert (m.bad_deletion == (m.lime_deletion_top3 <= m.random_deletion_3).astype(int)).all()
        assert len(abl[abl.model == model]) == 5
        assert len(sens[sens.model == model]) == 4
        assert len(bases[bases.model == model]) == 3
    for model, expected in {'LogisticRegression': (90, 38, 29, 9), 'XGBoost': (77, 67, 25, 42)}.items():
        r = ev[(ev.model == model) & (ev.partition == 'evaluation')].iloc[0]
        actual = tuple(int(r[k]) for k in ['flagged', 'bad', 'caught', 'missed'])
        assert actual == expected, f'{model} differs from saved run: {actual} != {expected}'
        print(f'PASS {model}: 300 evaluation clients; reviews, bad, caught, missed = {actual}')
    print('PASS calibration = 200 val clients, evaluation = 300 test clients, no overlap')
    print('PASS 4 threshold settings, 4 ablations, 3 baselines per model')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    check(parser.parse_args().output_dir)
