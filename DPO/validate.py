import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.training import validate
if __name__=='__main__': validate('DPO')
