"""Figures 7–8: fixed-model recording bootstrap and matched adaptation gains."""
from pathlib import Path
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--tables',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
if args.out.exists() and any(args.out.iterdir()):parser.error('--out must be absent or empty')
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=args.tables;F=args.out;F.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':320,'pdf.fonttype':42,'axes.labelsize':10})
ci=pd.read_csv(O/'bootstrap_ci.csv');COLORS=['#246A87','#B34D35']
def save(fig,name):
    fig.savefig(F/(name+'.png'),bbox_inches='tight',facecolor='white');fig.savefig(F/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(7.8,3.3),layout='constrained')
for i,ds in enumerate(['ETRI','uniD','exiD','pooled']):
    for j,unit in enumerate(['vehicle','physical_recording' if ds in ['exiD','pooled'] else 'recording']):
        r=ci[(ci.analysis=='A1')&(ci.eval_set==ds)&(ci.unit==unit)].iloc[0]
        ax.errorbar(r.point,3-i+(j-.5)*.18,xerr=[[r.point-r.ci_low],[r.ci_high-r.point]],fmt='o' if j==0 else 's',color=COLORS[j],capsize=3,label=['Vehicle resampling','Recording resampling (physical unit for exiD)'][j] if i==0 else None)
ax.axvline(.5,color='.4',ls='--',lw=.8);ax.set(yticks=[0,1,2,3],yticklabels=['Pooled targets','exiD','uniD','ETRI'],xlabel='ROC-AUC and conditional 95% interval',xlim=(.385,.60),ylim=(-.4,3.7),title='Source recording holdout: fixed seed-42 model')
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.22),fontsize=8,frameon=False,ncol=1);ax.grid(axis='x',alpha=.15)
save(fig,'fig7_recording_uncertainty')
fig,axs=plt.subplots(1,3,figsize=(8.0,3.4),layout='constrained')
for ax,ds,protocol in zip(axs,['ETRI','uniD','exiD'],['A2_recording','A2_recording','A2_physical_recording']):
    q=ci[(ci.analysis==protocol)&(ci.eval_set==ds)].sort_values('fraction')
    for i,r in enumerate(q.itertuples()):
        if ds=='ETRI':ax.plot(r.point,i,'x',color=COLORS[1],ms=6)
        else:ax.errorbar(r.point,i,xerr=[[r.point-r.ci_low],[r.ci_high-r.point]],fmt='o',color=COLORS[0],capsize=3)
    ax.axvline(0,color='.4',ls='--',lw=.8);ax.set(yticks=range(5),yticklabels=['10%','20%','40%','60%','100%'],xlabel='Continuation − target-only AUC',title=ds+(' (physical recordings)' if ds=='exiD' else ''),ylim=(-.5,5.1));ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
    ax.set_xticks([-.20,-.10,0] if ds=='ETRI' else [-.04,-.02,0] if ds=='uniD' else [-.01,-.005,0])
    if ds=='ETRI':ax.text(.5,.03,'Means only; interval degenerate\n1,003 / 2,000 valid resamples',transform=ax.transAxes,ha='center',va='bottom',fontsize=7)
    elif ds=='uniD':ax.text(.5,.03,'7 recording groups across seeds\n1,866 / 2,000 valid resamples',transform=ax.transAxes,ha='center',va='bottom',fontsize=7)
    else:ax.text(.5,.03,'59 physical groups across seeds\n2,000 / 2,000 valid resamples',transform=ax.transAxes,ha='center',va='bottom',fontsize=7)
axs[0].set_ylabel('Target pool fraction')
save(fig,'fig8_recording_gain')
print('Saved two data-driven figures with embedded-font PDF versions.')
