"""Reproduce the six R16 manuscript figures from their verified numeric inputs.

This script does not train a model or change the input data. Codex assisted its
development; the manuscript Methods describes the tool and verification scope.
"""
from pathlib import Path
import argparse, hashlib, io, json, platform, zipfile

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--tables',type=Path,required=True,help='Package verification directory')
parser.add_argument('--shap',type=Path,help='Optional authorized source-heldout SHAP npz; not supplied publicly')
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
if args.out.exists() and any(args.out.iterdir()):parser.error('--out must be absent or empty')
shap_bytes=args.shap.read_bytes() if args.shap else None
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
O=args.tables;F=args.out;F.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':320,'pdf.fonttype':42,'axes.labelsize':10})
COLORS=['#246A87','#B34D35','#54844B','#8461A1'];summary=json.loads((O/'summary.json').read_text())
def save(fig,name):
    fig.savefig(F/(name+'.png'),bbox_inches='tight',facecolor='white');fig.savefig(F/(name+'.pdf'),bbox_inches='tight');plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(8.0,3.1),layout='constrained')
for ax,fam,title in zip(axs,['xgb','lstm'],['a  Gradient-boosted trees','b  Bidirectional recurrent model']):
    dd=summary[fam];conds=['source_median_before','train_median_after'] if fam=='xgb' else ['before','after'];labels=['Before','After']
    for j,c in enumerate(conds):
        v=next(r for r in dd['summary'] if r['condition']==c and r['eval_set']=='in_domain');m=next(r for r in dd['macro'] if r['condition']==c)
        ax.bar(np.arange(2)+(j-.5)*.32,[v['mean'],m['mean']],.30,yerr=[v['sd'],m['sd']],capsize=3,label=labels[j],color=COLORS[j],zorder=3)
    ax.axhline(.5,c='.55',ls='--',lw=.8);ax.set(xticks=[0,1],xticklabels=['Source test','Target macro'],ylim=(0,1),ylabel='ROC-AUC',title=title);ax.grid(axis='y',alpha=.15);ax.legend(frameon=False,fontsize=8)
save(fig,'fig1_corrections')

met=pd.read_csv(O/'same_fit_metrics.csv').set_index('eval_set');ci=pd.read_csv(O/'same_fit_ci.csv');cal=pd.read_csv(O/'calibration_bins.csv')
fig,axs=plt.subplots(1,2,figsize=(8.0,3.4),layout='constrained')
sets=['ETRI','uniD','exiD','POOLED_NO_EMT'];names=['ETRI','uniD','exiD','Pooled targets']
for i,s in enumerate(sets):
    row=ci[(ci.eval_set==s)&(ci.unit=='vehicle')].iloc[0];a=met.loc[s,'auc'];axs[0].errorbar(a,3-i,xerr=[[a-row.lo],[row.hi-a]],fmt='o',color=COLORS[i],capsize=3)
axs[0].axvline(.5,c='.4',ls='--',lw=.8);axs[0].set(yticks=range(4),yticklabels=names[::-1],xlabel='ROC-AUC with vehicle bootstrap 95% interval',xlim=(.37,.57),title='a  Fixed source model at seed 42')
axs[1].plot([0,1],[0,1],'--',color='.5',lw=.8)
for i,s in enumerate(['in_domain','ETRI','uniD','exiD']):
    d=cal[(cal.eval_set==s)&(cal.n>0)];axs[1].plot(d.mean_pred,d.positive_fraction,'o-',ms=3,lw=1,color=COLORS[i],label='Source test' if s=='in_domain' else s)
axs[1].set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted probability',ylabel='Observed positive fraction',title='b  Reliability on the same predictions');axs[1].legend(frameon=False,fontsize=8)
save(fig,'fig2_samefit')

sh=pd.read_csv(O/'shap_recomputed.csv');top=sh[sh.eval_set=='source_heldout'].nlargest(12,'mean_abs_shap_log_odds').feature.tolist()
fig,axs=plt.subplots(1,2,figsize=(8.0,4.1),layout='constrained')
val=sh[sh.eval_set=='source_heldout'].set_index('feature').loc[top,'mean_abs_shap_log_odds']
axs[0].barh(np.arange(12),val,color=COLORS[0]);axs[0].set(yticks=range(12),yticklabels=top,xlabel='Mean absolute SHAP value (log-odds)',title='a  Source held-out rows');axs[0].invert_yaxis()
piv=sh.pivot(index='feature',columns='eval_set',values='mean_abs_shap_log_odds');cols=['source_heldout','ETRI','uniD','exiD'];a=(piv[cols]/piv[cols].sum()).loc[top].to_numpy();im=axs[1].imshow(a,aspect='auto',cmap='Blues',vmin=0)
axs[1].set(yticks=range(12),yticklabels=top,xticks=range(4),xticklabels=['Source','ETRI','uniD','exiD'],title='b  Share of total absolute attribution');axs[1].tick_params(axis='x',rotation=25);fig.colorbar(im,ax=axs[1],shrink=.7,label='Normalized share')
save(fig,'fig3_attribution')

if shap_bytes is not None:
    z=np.load(io.BytesIO(shap_bytes),allow_pickle=False);names=z['feature_names'].tolist();rng=np.random.RandomState(7)
    fig,ax=plt.subplots(figsize=(7.8,4.4),layout='constrained')
    for i,col in enumerate(top):
        j=names.index(col);x=z['values'][:,j];feat=z['features'][:,j];low,high=np.nanpercentile(feat,[1,99]);color=np.clip((feat-low)/(high-low if high>low else 1),0,1)
        ax.scatter(x,i+rng.uniform(-.25,.25,len(x)),c=color,cmap='coolwarm',norm=Normalize(0,1),s=2,alpha=.4,rasterized=True,linewidths=0)
    ax.axvline(0,c='.5',lw=.8);ax.set(yticks=range(12),yticklabels=top,xlabel='SHAP value (log-odds)',title='Source held-out attribution distribution');ax.invert_yaxis()
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,1),cmap='coolwarm'),ax=ax,shrink=.65,pad=.02);cb.set_ticks([0,1],labels=['Low','High']);cb.set_label('Feature value within each feature')
    save(fig,'fig4_shap_distribution')

else:
    print('SKIPPED Supplementary Fig. S3 regeneration: authorized sample-level SHAP/features required; fixed figure provided.')

adapt=pd.read_csv(O/'adaptation_no_emt.csv');paired=pd.read_csv(O/'paired_ci.csv')
fig,axs=plt.subplots(1,3,figsize=(8.0,3.1),layout='constrained')
for ax,ds in zip(axs,['ETRI','uniD','exiD']):
    for i,(cond,label) in enumerate([('continuation','Source continuation'),('target_only_target_median','Target only')]):
        d=adapt[(adapt.eval_set==ds)&(adapt.condition==cond)].sort_values('fraction');ax.errorbar(d.fraction*100,d['mean'],yerr=d.sd,marker='o',ms=3,capsize=2,label=label,color=COLORS[i],lw=1)
    ax.set(title=ds,xlabel='Target pool fraction (%)',xticks=[10,40,60,100],ylim=((.35,.8) if ds=='ETRI' else (.78,.94) if ds=='uniD' else (.97,1.0)));ax.grid(alpha=.15)
axs[0].set_ylabel('ROC-AUC (mean and seed SD)');axs[1].legend(loc='lower right',frameon=False,fontsize=7)
save(fig,'fig5_adaptation')

fig,axs=plt.subplots(1,3,figsize=(8.0,3.3),layout='constrained')
for ax,ds in zip(axs,['ETRI','uniD','exiD']):
    d=paired[paired.target==ds].sort_values('frac');ax.errorbar(d.gain,range(5),xerr=[d.gain-d.lo,d.hi-d.gain],fmt='o',ms=4,capsize=3,color=COLORS[0]);ax.axvline(0,c='.4',ls='--',lw=.8);ax.set(yticks=range(5),yticklabels=[f'{x:.0%}' for x in d.frac],title=ds);ax.invert_yaxis();ax.grid(axis='x',alpha=.15);ax.locator_params(axis='x',nbins=4)
axs[0].set_ylabel('Target pool fraction');fig.supxlabel('Continuation minus target-only ROC-AUC',fontsize=10);save(fig,'fig6_paired_gain')
print('Saved',len(list(F.glob('*.png'))),'figure pairs to',F)

versions={'python':platform.python_version(),'numpy':np.__version__,
          'pandas':pd.__version__,'matplotlib':matplotlib.__version__}
inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(O.iterdir()) if p.suffix in {'.csv','.json'}}
manifest={'versions':versions,'shap_sha256':hashlib.sha256(shap_bytes).hexdigest() if shap_bytes is not None else None,
          'source_shap_npz_sha256':hashlib.sha256(shap_bytes).hexdigest() if shap_bytes is not None else None,
          'table_input_sha256':inputs,
          'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'png_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(F.glob('*.png'))}}
(F/'plot_run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
