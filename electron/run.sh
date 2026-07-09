#!/usr/bin/env bash
# NavierTwin 데스크톱 앱 실행 (WSL/WSLg).
#
# npm(윈도우 interop) 이 PATH 에 잡혀도 문제없도록 설치된 Linux electron
# 바이너리를 직접 실행한다. WSL 에서는 --no-sandbox 가 필요하다.
set -e
cd "$(dirname "$0")"

if [ ! -x node_modules/electron/dist/electron ]; then
  echo "electron 이 설치되어 있지 않습니다. 먼저 'npm install' 을 실행하세요." >&2
  echo "  (Linux Node 필요: nvm use --lts && npm install)" >&2
  exit 1
fi

exec ./node_modules/electron/dist/electron . --no-sandbox "$@"
