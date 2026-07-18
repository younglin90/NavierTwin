"""NavierTwin 프로젝트 루트 실행 진입점.

``python main.py`` 또는 ``python main.py --gui`` 로 실행한다.
패키지 설치 없이 개발 모드에서 바로 실행할 때 사용한다.

Note:
    패키지가 설치된 환경에서는 ``naviertwin`` CLI 명령을 사용하는 것을 권장한다.
"""

from pathlib import Path
import sys

# 개발 모드에서 패키지 설치 없이 실행할 수 있도록 src를 경로에 추가한다.
_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from naviertwin.main import main

if __name__ == "__main__":
    main()
