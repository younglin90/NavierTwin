// NavierTwin 데스크톱 셸 (Electron).
//
// Linux 에서는 Python 서버를 직접 실행한다. Windows 에서는 wsl.exe 로 같은
// Linux 서버를 실행한다. 준비되면 메인 창에 로드하고 종료 시 자식을 정리한다.

const { app, BrowserWindow, dialog } = require("electron");
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");
const {
  findFreePort,
  stopChildProcess,
  waitForServer,
} = require("./server-process");
const { resolveServerLaunch } = require("./server-launch");

const REPO_ROOT = path.resolve(__dirname, "..");
const DEFAULT_PORT = 8877;
const SMOKE = process.argv.includes("--smoke");

// __dirname 은 패키징된 빌드에서 읽기 전용 app.asar 내부를 가리킨다 — 진단
// 로그(boot.log/server.log)를 그 안에 쓰려 하면 packaged 빌드에서 조용히
// 실패하거나(캐치되지 않은 스트림 오류) 최악의 경우 프로세스가 죽는다.
// 항상 쓰기 가능한 사용자 데이터 디렉터리에 기록한다.
const LOG_DIR = app.getPath("userData");
try {
  fs.mkdirSync(LOG_DIR, { recursive: true });
} catch (e) {
  /* ignore */
}

// WSLg 에서 GPU 초기화가 멈추는 것을 방지 (소프트웨어 렌더).
app.disableHardwareAcceleration();
app.commandLine.appendSwitch("disable-gpu");
app.commandLine.appendSwitch("disable-dev-shm-usage");
app.commandLine.appendSwitch("no-sandbox");

// 부팅 진단 마커 (main.js 가 실제 로드/실행되는지 확인용).
try {
  fs.writeFileSync(
    path.join(LOG_DIR, "boot.log"),
    `main.js loaded, smoke=${SMOKE}, pid=${process.pid}\n`
  );
} catch (e) {
  /* ignore */
}

let serverProc = null;
let serverPort = DEFAULT_PORT;
let mainWindow = null;
let splashWindow = null;

// ── 서버 기동 ───────────────────────────────────────────────────────
function launchServer(port) {
  const launch = resolveServerLaunch({
    platform: process.platform,
    repoRoot: REPO_ROOT,
    port,
    env: process.env,
  });
  return spawn(launch.command, launch.args, launch.options);
}

// ── 창 ──────────────────────────────────────────────────────────────
// naviertwin/web/theme.py 의 BACKGROUND 와 동일 (단일 팔레트 출처 — 색을 바꾸면
// 양쪽 다 갱신할 것). 창 배경이 다르면 페이지 로드 전/후 플래시가 어긋나 보인다.
const THEME_BACKGROUND = "#1b2739";

function createSplash() {
  splashWindow = new BrowserWindow({
    width: 460,
    height: 300,
    frame: false,
    resizable: false,
    transparent: false,
    backgroundColor: THEME_BACKGROUND,
    show: true,
    webPreferences: { contextIsolation: true },
  });
  splashWindow.loadFile(path.join(__dirname, "splash.html"));
}

function createMainWindow(port) {
  mainWindow = new BrowserWindow({
    width: 1500,
    height: 950,
    minWidth: 1200,
    minHeight: 780,
    backgroundColor: THEME_BACKGROUND,
    autoHideMenuBar: true,
    show: false,
    icon: path.join(__dirname, "assets", "icon.png"),
    title: "NavierTwin",
    webPreferences: {
      contextIsolation: true,
      preload: path.join(__dirname, "preload.js"),
    },
  });
  mainWindow.loadURL(`http://127.0.0.1:${port}/index.html`);
  mainWindow.once("ready-to-show", () => {
    if (splashWindow) {
      splashWindow.close();
      splashWindow = null;
    }
    mainWindow.show();
  });
  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

// ── smoke: 창 캡처 후 종료 ──────────────────────────────────────────
function runSmoke(port) {
  const win = new BrowserWindow({
    width: 1500,
    height: 950,
    show: false,
    backgroundColor: THEME_BACKGROUND,
    webPreferences: { contextIsolation: true },
  });
  win.loadURL(`http://127.0.0.1:${port}/index.html`);
  win.webContents.on("did-finish-load", () => {
    setTimeout(async () => {
      try {
        const img = await win.webContents.capturePage();
        fs.writeFileSync(
          path.join(LOG_DIR, "smoke.png"),
          img.toPNG()
        );
        console.log("SMOKE_OK smoke.png written");
      } catch (e) {
        console.error("SMOKE_FAIL", e);
      } finally {
        app.quit();
      }
    }, 3500);
  });
}

// ── 정리 ────────────────────────────────────────────────────────────
function stopServer() {
  stopChildProcess(serverProc, { timeoutMs: 2000 });
}

// ── 부트스트랩 ──────────────────────────────────────────────────────
async function bootstrap() {
  try {
    fs.appendFileSync(path.join(LOG_DIR, "boot.log"), "bootstrap start\n");
  } catch (e) {
    /* ignore */
  }
  try {
    serverPort = await findFreePort(DEFAULT_PORT);
  } catch (e) {
    serverPort = DEFAULT_PORT;
  }

  if (!SMOKE) createSplash();

  try {
    serverProc = launchServer(serverPort);
  } catch (error) {
    if (splashWindow) splashWindow.close();
    dialog.showErrorBox("서버 설정 오류", error.message);
    app.quit();
    return;
  }
  const logPath = path.join(LOG_DIR, "server.log");
  const logStream = fs.createWriteStream(logPath, { flags: "w" });
  // Without an "error" listener a stream write failure (e.g. LOG_DIR not
  // writable) is an uncaught exception that kills the main process.
  logStream.on("error", () => {
    /* ignore — logging is best-effort */
  });
  if (serverProc.stdout) serverProc.stdout.pipe(logStream);
  if (serverProc.stderr) serverProc.stderr.pipe(logStream);
  serverProc.on("exit", (code) => {
    console.log(`server exited: ${code}`);
  });

  try {
    fs.appendFileSync(
      path.join(LOG_DIR, "boot.log"),
      `server spawned on port ${serverPort}, waiting…\n`
    );
  } catch (e) {
    /* ignore */
  }
  const ready = await waitForServer(serverPort, 60000);
  try {
    fs.appendFileSync(path.join(LOG_DIR, "boot.log"), `server ready=${ready}\n`);
  } catch (e) {
    /* ignore */
  }
  if (!ready) {
    let tail = "";
    try {
      tail = fs.readFileSync(logPath, "utf8").split("\n").slice(-20).join("\n");
    } catch (e) {
      /* ignore */
    }
    if (splashWindow) splashWindow.close();
    dialog.showErrorBox(
      "서버 기동 실패",
      `NavierTwin 웹 서버가 60초 내에 준비되지 않았습니다.\n\n${tail}`
    );
    app.quit();
    return;
  }

  if (SMOKE) {
    runSmoke(serverPort);
  } else {
    createMainWindow(serverPort);
  }
}

app.whenReady().then(bootstrap);

app.on("before-quit", stopServer);
app.on("window-all-closed", () => {
  stopServer();
  app.quit();
});
