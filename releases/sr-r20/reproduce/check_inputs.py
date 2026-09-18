"""No training: fail explicitly when authorized H3 feature inputs are absent."""
from pathlib import Path
import argparse,sys
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',type=Path,required=True);a=p.parse_args();names=['highD','NGSIM','MiTra','ETRI','uniD','exiD'];missing=[str(a.data_root/'processed'/f'{d}_gt_3s.csv') for d in names if not (a.data_root/'processed'/f'{d}_gt_3s.csv').is_file()]
if missing:sys.exit('Required authorized inputs missing. See DATA_ACCESS.md; no scores generated.\n'+'\n'.join(missing))
print('Six required H3 files present. This checks presence only, not licensing, semantics, hashes or training reproducibility.')
