
"""COSC 4V88 Project 5 Part C: reproducible explanation reliability flag.

Author of the method: Ayaan Alam Biabani Feroz.
v2 (re-run by Shuo on the expanded explanation sample; method unchanged):
  - Uses the shared 500-client explanation sample (explain_sample.csv):
    200 VALIDATION clients are used for calibration (signal scalers and cutoffs),
    300 TEST clients are used only for evaluation. This replaces the earlier
    50/50 split of 100 test clients, so calibration and evaluation no longer
    both come from the test set.
  - Validation predictions are read from predictions_val.csv.
  - Added workload_matched.csv: every signal compared at the same number of reviews.
  - Demo clients: the Part B demo clients (17775 success, 6391 failure) plus the
    first passing and the highest-scoring flagged XGBoost client in evaluation.

The score uses four signal families: instability, SHAP/LIME disagreement,
LIME local-fit failure, and borderline prediction. Its "bad" criterion is
deletion-test failure (LIME top-3 causes <= random top-3 prediction change)
and is NOT included in the score. Ablation review cutoffs target the same
workload on the CALIBRATION clients only; evaluation workloads are not matched.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta

SIGNALS = ['instability', 'disagreement', 'local_fit', 'borderline']
MODELS = {'LogisticRegression':'lr','XGBoost':'xgb'}
SEED = 88
REVIEW_FRACTION = 0.30
CAL_LABEL = 'calibration_200_val_clients'
EVAL_LABEL = 'evaluation_300_test_clients'
PARTB_DEMO_IDS = [17775, 6391]


def read_unique(path: Path, keys: list[str]):
    df = pd.read_csv(path)
    assert not df.duplicated(keys).any(), f'duplicate key in {path} for {keys}'
    return df


def load_records(root: Path):
    r = root / '03_Results'; a=r/'tables/PartA_Step2_tables'; b=r/'tables/PartB_tables'; ap=r/'tables/PartA_Step1_tables'
    settings=json.loads((r/'config.json').read_text())
    sample=read_unique(b/'explain_sample.csv',['ID'])[['ID','default','split']]
    stability={
        'LogisticRegression':read_unique(b/'lime_stability_lr.csv',['ID'])[['ID','avg_jaccard','avg_sign_flip']],
        'XGBoost':read_unique(b/'lime_stability_xgb.csv',['ID'])[['ID','avg_jaccard','avg_sign_flip']]
    }
    agreement=read_unique(b/'shap_lime_agreement.csv',['ID'])
    fidelity=read_unique(a/'fidelity_per_client.csv',['ID','model'])
    predictions=pd.concat([pd.read_csv(ap/'predictions_val.csv'),pd.read_csv(ap/'predictions_test.csv')],ignore_index=True)
    assert not predictions.duplicated('ID').any()
    data=pd.concat([pd.read_csv(root/'01_Data/processed/val.csv'),pd.read_csv(root/'01_Data/processed/test.csv')],ignore_index=True)
    assert not data.duplicated('ID').any()
    n=len(sample)
    chunks=[]
    for model,suffix in MODELS.items():
        d=sample.rename(columns={'split':'data_split'})
        d=d.merge(stability[model],on='ID',validate='one_to_one')
        d=d.merge(agreement[['ID','agreement_jaccard_'+suffix]],on='ID',validate='one_to_one')
        d=d.merge(fidelity[fidelity.model==model].drop(columns=['default','split','row_index','test_row_index'],errors='ignore'),on='ID',validate='one_to_one')
        d=d.merge(predictions[['ID','proba_'+model]],on='ID',validate='one_to_one')
        d=d.merge(data[['ID','SEX','EDUCATION','AGE']],on='ID',validate='one_to_one')
        assert len(d)==n and d.notnull().all().all(), f'missing data in {model} join'
        d=d.rename(columns={'agreement_jaccard_'+suffix:'agreement_jaccard','proba_'+model:'probability'})
        d['model']=model
        d['decision_threshold']=float(settings['thresholds_chosen_on_validation'][model])
        d['instability_raw']=0.70*(1-d.avg_jaccard)+0.30*d.avg_sign_flip
        d['disagreement_raw']=1-d.agreement_jaccard
        d['local_fit_raw']=np.log1p(d.lime_relative_wmse.clip(lower=0))
        d['borderline_raw']=np.exp(-(d.probability-d.decision_threshold).abs()/0.10)
        d['bad_deletion']= (d.lime_deletion_top3<=d.random_deletion_3).astype(int)
        chunks.append(d)
    full=pd.concat(chunks,ignore_index=True)
    assert full.ID.nunique()==n and len(full)==2*n
    return full,settings


def empirical_percentiles(values, cal):
    values=np.asarray(values,dtype=float);cal=np.sort(np.asarray(cal,dtype=float))
    assert len(cal)>0 and np.isfinite(values).all() and np.isfinite(cal).all()
    return np.searchsorted(cal,values,side='right')/len(cal)


def evaluate_stats(d,flag_col='flag'):
    flag=d[flag_col].astype(bool).to_numpy();bad=d.bad_deletion.astype(bool).to_numpy()
    n=len(d);tp=int(np.sum(flag & bad)); fp=int(np.sum(flag & ~bad));fn=int(np.sum(~flag & bad));tn=int(np.sum(~flag & ~bad))
    def divide(n,d):return float(n/d) if d else np.nan
    def cp(success,total):
        if total==0:return (np.nan,np.nan)
        return (0 if success==0 else float(beta.ppf(.025,success,total-success+1)),
                1 if success==total else float(beta.ppf(.975,success+1,total-success)))
    rec_ci=cp(tp,tp+fn)
    return {'n':n,'flagged':tp+fp,'bad':tp+fn,'caught':tp,'missed':fn,
            'false_alarms':fp,'safe_unflagged':tn,
            'review_rate':divide(tp+fp,n), 'bad_rate_flagged':divide(tp,tp+fp),
            'bad_rate_unflagged':divide(fn,fn+tn),'precision':divide(tp,tp+fp),
            'recall':divide(tp,tp+fn),'recall_ci_low':rec_ci[0], 'recall_ci_high':rec_ci[1],
            'enrichment':divide(divide(tp,tp+fp),divide(tp+fn,n))}


def run(root: Path, out: Path):
    out.mkdir(parents=True,exist_ok=True)
    d,config=load_records(root)
    train_ids=set(d.loc[d.data_split=='val','ID'].tolist());holdout_ids=set(d.loc[d.data_split=='test','ID'].tolist())
    d['split']=np.where(d.ID.isin(train_ids),CAL_LABEL,EVAL_LABEL)
    assert not train_ids & holdout_ids
    d['flag']=0;d['score']=np.nan
    ablations=[];sensitivities=[];baselines=[];summaries=[];calibration_info={}
    for model in MODELS:
        m=d.model==model;cal=d.loc[m & d.ID.isin(train_ids)].copy()
        for signal in SIGNALS:
            col=signal+'_risk';raw=signal+'_raw'
            d.loc[m,col]=empirical_percentiles(d.loc[m,raw],cal[raw])
        cal=d.loc[m & d.ID.isin(train_ids)].copy()
        ev=d.loc[m & d.ID.isin(holdout_ids)].copy()
        def combined(z,keep):return z[[s+'_risk' for s in keep]].mean(axis=1).to_numpy()
        for data in (cal,ev):
            data.loc[:,'score']=combined(data,SIGNALS)
        cutoff=float(np.quantile(cal.score,1-REVIEW_FRACTION,method='higher'))
        d.loc[m,'score']=combined(d.loc[m],SIGNALS)
        d.loc[m,'flag']=(d.loc[m,'score']>=cutoff).astype(int)
        cal=d.loc[m & d.ID.isin(train_ids)].copy();ev=d.loc[m & d.ID.isin(holdout_ids)].copy()
        calibration_info[model]={'cutoff':cutoff,'calibration_n':len(cal),'evaluation_n':len(ev),
            'calibration_review_rate':float(cal.flag.mean()),
            'calibration_bad_count':int(cal.bad_deletion.sum()),
            'evaluation_bad_count':int(ev.bad_deletion.sum())}
        for partition,rows in [('calibration',cal),('evaluation',ev)]:
            summaries.append({'model':model,'partition':partition,'method':'four_signal','cutoff':cutoff,**evaluate_stats(rows)})
        for fraction in [0.20,0.30,0.40,0.50]:
            threshold=float(np.quantile(cal.score,1-fraction,method='higher'))
            flags=(ev.score>=threshold).astype(int)
            e=ev.assign(flag_sensitivity=flags)
            sensitivities.append({'model':model,'target_review_fraction':fraction,'calibration_score_cutoff':threshold,**evaluate_stats(e,'flag_sensitivity')})
        for remove in [None]+SIGNALS:
            keep=[s for s in SIGNALS if s!=remove]
            sc=combined(cal,keep);se=combined(ev,keep)
            threshold=float(np.quantile(sc,1-REVIEW_FRACTION,method='higher'))
            rows=ev.assign(flag_ablation=(se>=threshold).astype(int))
            ablations.append({'model':model,'ablation':'all_four' if remove is None else 'drop_'+remove,
                              'cutoff':threshold,**evaluate_stats(rows,'flag_ablation')})
        for name,signal in [('borderline_only','borderline'),('instability_only','instability'),('local_fit_only','local_fit')]:
            threshold=float(np.quantile(cal[signal+'_risk'],1-REVIEW_FRACTION,method='higher'))
            rows=ev.assign(flag_base=(ev[signal+'_risk']>=threshold).astype(int))
            baselines.append({'model':model,'method':name,'cutoff':threshold,**evaluate_stats(rows,'flag_base')})
    # Workload-matched comparison (added in v2): rank the evaluation clients by each single
    # signal, or by the four-signal score, and send exactly as many to review as the
    # four-signal flag did. This uses the evaluation review count, so it is a fair-comparison
    # check, not a deployable rule.
    matched=[]
    for model in MODELS:
        ev=d[(d.model==model)&(d.split==EVAL_LABEL)]
        k=int(ev.flag.sum())
        rankers=[('four_signal','score')]+[(s+'_only',s+'_raw') for s in SIGNALS]
        for name,col in rankers:
            top=ev.sort_values([col,'ID'],ascending=[False,True]).head(k).ID
            rows=ev.assign(flag_matched=ev.ID.isin(top).astype(int))
            matched.append({'model':model,'method':name,'reviews':k,**evaluate_stats(rows,'flag_matched')})
        bad=int(ev.bad_deletion.sum())
        matched.append({'model':model,'method':'random_expected','reviews':k,'n':len(ev),'flagged':k,'bad':bad,
                        'caught':round(k*bad/len(ev),1),'recall':k/len(ev)})
    pd.DataFrame(matched).to_csv(out/'workload_matched.csv',index=False)
    d.sort_values(['model','ID']).to_csv(out/'client_flags.csv',index=False)
    pd.DataFrame(summaries).to_csv(out/'flag_evaluation.csv',index=False)
    pd.DataFrame(ablations).to_csv(out/'ablation.csv',index=False)
    pd.DataFrame(sensitivities).to_csv(out/'threshold_sensitivity.csv',index=False)
    pd.DataFrame(baselines).to_csv(out/'baselines.csv',index=False)
    xe=d[(d.model=='XGBoost') & (d.split==EVAL_LABEL)]
    first_pass=int(xe[(xe.flag==0)&(xe.bad_deletion==0)].sort_values(['score','ID']).iloc[0].ID)
    top_flag=int(xe[(xe.flag==1)&(xe.bad_deletion==1)].sort_values(['score','ID'],ascending=[False,True]).iloc[0].ID)
    demo_ids=PARTB_DEMO_IDS+[first_pass,top_flag]
    demo=d.loc[d.ID.isin(demo_ids),['ID','model','probability','decision_threshold','avg_jaccard','avg_sign_flip','agreement_jaccard','lime_relative_wmse','lime_faithful_vs_random','bad_deletion','score','flag','split']].sort_values(['ID','model'])
    demo.to_csv(out/'demo_cases.csv',index=False)
    meta={'seed':SEED,'version':'v2 (500-client sample; method unchanged)',
          'note':'Calibration on the 200 validation clients, evaluation on the 300 test clients of the shared explanation sample.',
          'models':list(MODELS),'review_target':REVIEW_FRACTION,'signals':SIGNALS,
          'score_definition':'arithmetic mean of four calibration-percentile transformed risks; instability_raw=.7*(1-jaccard)+.3*sign_flip; disagreement_raw=1-jaccard; local_fit_raw=log(1+lime_relative_wmse); borderline_raw=exp(-|p-model_decision_threshold|/0.10)',
          'bad_explanation_definition':'lime_deletion_top3 <= random_deletion_3; this outcome does not enter the score',
          'calibration':calibration_info,'first_pass_client_XGBoost':first_pass,'top_flagged_bad_client_XGBoost':top_flag,
          'partb_demo_ids':PARTB_DEMO_IDS,
          'data_provenance':'PartA_Step2 v2, PartB v2, PartA_Step1 val+test predictions'}
    (out/'method_config.json').write_text(json.dumps(meta,indent=2))
    print('\nFLAG EVALUATION\n',pd.DataFrame(summaries).to_string(index=False))
    print('\nABLATION\n',pd.DataFrame(ablations).to_string(index=False))
    print('\nTHRESHOLD SENSITIVITY\n',pd.DataFrame(sensitivities).to_string(index=False))
    print('\nBASELINES\n',pd.DataFrame(baselines).to_string(index=False))
    print('\nWORKLOAD-MATCHED\n',pd.DataFrame(matched)[['model','method','reviews','caught','bad','recall']].to_string(index=False))
    print('\nDEMO\n',demo.to_string(index=False))
    return meta

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--project-root',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();run(args.project_root,args.output_dir)
