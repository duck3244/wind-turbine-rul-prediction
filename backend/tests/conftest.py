"""루트 모듈을 import 할 수 있도록 PYTHONPATH를 보정."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
