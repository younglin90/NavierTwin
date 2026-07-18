// NavierTwin 데스크톱 셸 (Electron).
//
// WSL(Linux) 안에서 실행되어 WSLg 로 Windows 데스크톱에 창을 띄운다. trame 웹
// 서버(python -m naviertwin.main web)를 자식 프로세스로 기동하고, 준비되면
// 메인 창에 로드한다. 종료 시 서버를 정리한다.
//
// 미래 Windows-native 빌드를 위해 서버 기동은 launchServer(mode) 로 분리했다:
//   - 'local'(현재): electron 이 WSL 안 → python3 직접 spawn
//   - 'wsl'(스텁): electron 이 Windows → wsl.exe 경유 spawn (주석 참고)

const { app, BrowserWindow, dialog } = require("electron");
const { spawn } = require("child_process");
const http = require("http");
const net = require("net");
const path = require("path");
const fs = require("fs");

const REPO_ROOT = path.resolve(__dirname, "..");
const DEFAULT_PORT = 8877;
const SMOKE = process.argv.includes("--smoke");

// WSLg 에서 GPU 초기화가 멈추는 것을 방지 (소프트웨어 렌더).
app.disableHardwareAcceleration();
app.commandLine.appendSwitch("disable-gpu");
app.commandLine.appendSwitch("disable-dev-shm-usage");
app.commandLine.appendSwitch("no-sandbox");

// 부팅 진단 마커 (main.js 가 실제 로드/실행되는지 확인용).
try {
  fs.writeFileSync(
    path.join(__dirname, "boot.log"),
    `main.js loaded, smoke=${SMOKE}, pid=${process.pid}\n`
  );
} catch (e) {
  /* ignore */
}

let serverProc = null;
let serverPort = DEFAULT_PORT;
let mainWindow = null;
let splashWindow = null;

// ── 포트 ────────────────────────────────────────────────────────────
function isPortFree(port) {
  return new Promise((resolve) => {
    const srv = net.createServer();
    srv.once("error", () => resolve(false));
    srv.once("listening", () => srv.close(() => resolve(true)));
    srv.listen(port, "127.0.0.1");
  });
}

async function findFreePort(start) {
  for (let p = start; p < start + 100; p += 1) {
    // eslint-disable-next-line no-await-in-loop
    if (await isPortFree(p)) return p;
  }
  throw new Error("사용 가능한 포트를 찾지 못했습니다.");
}

// ── 서버 기동 ───────────────────────────────────────────────────────
function launchServer(mode, port) {
  const args = [
    "-m",
    "naviertwin.main",
    "web",
    "--host",
    "127.0.0.1",
    "--port",
    String(port),
    "--no-browser",
  ];
  const env = {
    ...process.env,
    QT_QPA_PLATFORM: "offscreen",
    PYVISTA_OFF_SCREEN: "true",
    PYTHONPATH: "src",
  };
  if (mode === "wsl") {
    // Windows-native 빌드용 스텁: electron 이 Windows 에서 돌 때.
    //   return spawn("wsl", ["-d", "ubuntu", "--", "bash", "-lc",
    //     `cd ~/work/claude_code/NavierTwin && QT_QPA_PLATFORM=offscreen ` +
    //     `PYVISTA_OFF_SCREEN=true PYTHONPATH=src python3 ${args.join(" ")}`]);
    throw new Error("wsl 모드는 아직 구현되지 않았습니다 (현재 local 전용).");
  }
  return spawn("python3", args, { cwd: REPO_ROOT, env });
}

// ── 헬스 체크 ───────────────────────────────────────────────────────
function checkHealth(port) {
  return new Promise((resolve) => {
    const req = http.get(
      { host: "127.0.0.1", port, path: "/index.html", timeout: 800 },
      (res) => {
        res.resume();
        resolve(res.statusCode === 200);
      }
    );
    req.on("error", () => resolve(false));
    req.on("timeout", () => {
      req.destroy();
      resolve(false);
    });
  });
}

async function waitForServer(port, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    // eslint-disable-next-line no-await-in-loop
    if (await checkHealth(port)) return true;
    // eslint-disable-next-line no-await-in-loop
    await new Promise((r) => setTimeout(r, 500));
  }
  return false;
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
          path.join(__dirname, "smoke.png"),
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
  if (serverProc && !serverProc.killed) {
    try {
      serverProc.kill("SIGTERM");
    } catch (e) {
      /* ignore */
    }
    // 2초 후에도 살아있으면 강제 종료 + 안전망(pkill).
    setTimeout(() => {
      try {
        if (serverProc && !serverProc.killed) serverProc.kill("SIGKILL");
      } catch (e) {
        /* ignore */
      }
      try {
        spawn("pkill", ["-f", "naviertwin.main web"]);
      } catch (e) {
        /* ignore */
      }
    }, 2000);
  }
}

// ── 부트스트랩 ──────────────────────────────────────────────────────
async function bootstrap() {
  try {
    fs.appendFileSync(path.join(__dirname, "boot.log"), "bootstrap start\n");
  } catch (e) {
    /* ignore */
  }
  try {
    serverPort = await findFreePort(DEFAULT_PORT);
  } catch (e) {
    serverPort = DEFAULT_PORT;
  }

  if (!SMOKE) createSplash();

  serverProc = launchServer("local", serverPort);
  const logPath = path.join(__dirname, "server.log");
  const logStream = fs.createWriteStream(logPath, { flags: "w" });
  if (serverProc.stdout) serverProc.stdout.pipe(logStream);
  if (serverProc.stderr) serverProc.stderr.pipe(logStream);
  serverProc.on("exit", (code) => {
    console.log(`server exited: ${code}`);
  });

  try {
    fs.appendFileSync(
      path.join(__dirname, "boot.log"),
      `server spawned on port ${serverPort}, waiting…\n`
    );
  } catch (e) {
    /* ignore */
  }
  const ready = await waitForServer(serverPort, 60000);
  try {
    fs.appendFileSync(path.join(__dirname, "boot.log"), `server ready=${ready}\n`);
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
