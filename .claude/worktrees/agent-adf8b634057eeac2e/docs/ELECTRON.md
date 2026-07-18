# NavierTwin 데스크톱 앱 (Electron)

trame 웹 GUI 를 감싸는 Electron 데스크톱 셸. 서버는 **WSL(Linux Python)** 전용
(native kernel 이 Linux `.so`)이므로, Electron 도 **WSL 안에서 실행**하고 **WSLg**
로 Windows 데스크톱에 창을 띄운다. 이렇게 하면 `\\wsl.localhost` UNC 경로에서
Windows npm 을 돌리는 문제를 피하고, 서버를 단순한 Linux 자식 프로세스로 기동한다.

## 구조

```
electron/
├── package.json      # electron devDependency + start/smoke 스크립트
├── main.js           # 서버 spawn + 헬스 폴링 + 창/스플래시 + 종료 정리
├── preload.js        # 최소 (contextIsolation 유지)
├── splash.html       # 다크 스플래시 (기동 중 파동 애니메이션)
├── run.sh            # Linux electron 바이너리 직접 실행 (npm PATH 이슈 우회)
└── assets/icon.png   # 앱 아이콘
```

`main.js` 흐름: 빈 포트 탐색(8877~) → `python3 -m naviertwin.main web` 를 repo
루트에서 `PYTHONPATH=src`, `QT_QPA_PLATFORM=offscreen`, `PYVISTA_OFF_SCREEN=true`
로 spawn → 스플래시 표시 → `http://127.0.0.1:PORT/index.html` 200 폴링(최대 60s,
torch 첫 import 가 느림) → 메인 창 로드 → 종료 시 서버 `SIGTERM`→`SIGKILL`(+`pkill`
안전망). WSLg GPU 초기화 hang 회피를 위해 하드웨어 가속을 끈다.

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

## 향후 (범위 외)

- `electron-builder` 로 배포 패키지(AppImage/NSIS) 생성
- Windows-native Electron 모드: `main.js` 의 `launchServer('wsl')` 스텁을 구현해
  `wsl.exe -d ubuntu -- …` 로 서버를 기동(현재는 `'local'` 전용)
