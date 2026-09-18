"""Verify and regenerate seven aggregate figure pairs in a fresh output directory."""
from pathlib import Path
import argparse,subprocess,sys,json
from verify_public import verify
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
if a.out.exists():p.error('--out must not exist; choose a new directory')
result=verify(root);a.out.mkdir(parents=True)
for stage,script in [('R16','plot_vehicle_public.py'),('R17','plot_recording_results.py')]:subprocess.run([sys.executable,str(root/'reproduce'/script),'--tables',str(root/'aggregates'/stage),'--out',str(a.out/stage)],check=True)
result.update(regenerated_figure_pairs=7,fixed_only='Supplementary S3: figures/fig4_shap_distribution; authorized arrays required to regenerate');(a.out/'PUBLIC_REPLAY.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
