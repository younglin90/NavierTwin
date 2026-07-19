# NavierTwin 데스크톱 앱 (Electron)

trame 웹 GUI 를 감싸는 Electron 데스크톱 셸. Linux/WSLg에서는 Python 서버를
직접 실행한다. Windows-native Electron에서는 `wsl.exe`로 Linux 서버를 실행한다.
native kernel은 두 모드 모두 WSL의 Linux `.so`를 사용한다.

## 구조

```
electron/
├── package.json      # electron devDependency + start/smoke 스크립트
├── main.js           # 서버 spawn + 헬스 폴링 + 창/스플래시 + 종료 정리
├── server-launch.js  # Linux/Windows-WSL 실행 명령 생성
├── server-process.js # 포트, 헬스체크, 자식 종료
├── preload.js        # 최소 (contextIsolation 유지)
├── splash.html       # 다크 스플래시 (기동 중 파동 애니메이션)
├── run.sh            # Linux electron 바이너리 직접 실행 (npm PATH 이슈 우회)
└── assets/icon.png   # 앱 아이콘
```

`main.js` 흐름: 빈 포트 탐색(8877~) → `python3 -m naviertwin.main web` 를 repo
루트에서 `PYTHONPATH=src`, `QT_QPA_PLATFORM=offscreen`, `PYVISTA_OFF_SCREEN=true`
로 spawn → 스플래시 표시 → `http://127.0.0.1:PORT/index.html` 200 폴링(최대 60s,
torch 첫 import 가 느림) → 메인 창 로드 → 종료 시 추적 중인 자식 서버만
`SIGTERM` 후 `SIGKILL`한다. 다른 NavierTwin 인스턴스나 개발 서버는 종료하지
않는다. WSLg GPU 초기화 hang 회피를 위해 하드웨어 가속을 끈다.

Windows-native 실행에서 저장소가 `\\wsl.localhost\\배포판\\...` 아래면 배포판과
Linux 경로를 자동 추출한다. Windows 드라이브에 Electron 코드가 있으면 다음 환경
변수를 지정한다.

```powershell
$env:NAVIER_TWIN_WSL_DISTRO = "Ubuntu"
$env:NAVIER_TWIN_WSL_PROJECT = "/home/user/work/NavierTwin"
npm start
```

명령은 셸 문자열 없이 `wsl.exe` 인자 배열로 전달한다.

## 선행조건

- **WSL2 + WSLg** (`echo $WAYLAND_DISPLAY` → `wayland-0`)
- **Linux Node LTS** (nvm 권장 — WSL 의 `node`/`npm` 이 Windows 바이너리로 잡히면
  `electron` npm 이 잘못된 플랫폼을 받는다):
  ```bash
  nvm install --lts && nvm use --lts
  ```
- Electron Linux 런타임 lib (Ubuntu 는 대개 이미 있음): libgtk-3, libnss3,
  libasound2, libatk-bridge2.0, libgbm1, libxkbcommon0.

## 설치 & 실행

```bash
cd ~/work/claude_code/NavierTwin/electron
nvm use --lts        # Linux Node 보장
npm install          # Linux electron 다운로드
./run.sh             # 앱 실행 (WSLg 창이 Windows 데스크톱에 뜸)
#   또는: npm start  (Linux node 가 PATH 우선일 때)
```

기동 스모크(창을 캡처해 `electron/smoke.png` 저장 후 종료):

```bash
./node_modules/electron/dist/electron . --smoke --no-sandbox
```

## 배포 패키지

```bash
cd electron
npm ci
npm run dist:linux
npm run dist:win
```

Linux는 AppImage, Windows는 설치 위치를 선택할 수 있는 NSIS 설치 파일을
`electron/dist/`에 생성한다. GitHub Actions의 `Electron package` workflow도
두 플랫폼 산출물을 artifact로 보존한다. 코드 서명 인증서가 없는 개발 빌드는
서명되지 않는다. Windows 드라이브 설치본은 서버 위치를 찾기 위해
`NAVIER_TWIN_WSL_PROJECT`가 필요하다.
