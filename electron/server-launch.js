"use strict";

const path = require("path");

const DEFAULT_DISTRO = "Ubuntu";

function uncToWslPath(repoRoot) {
  const normalized = String(repoRoot).replaceAll("/", "\\");
  const match = normalized.match(
    /^\\\\(?:wsl\.localhost|wsl\$)\\([^\\]+)\\(.+)$/i
  );
  if (!match) return null;
  return {
    distro: match[1],
    projectRoot: `/${match[2].replaceAll("\\", "/")}`,
  };
}

function serverArgs(port) {
  return [
    "-m",
    "naviertwin.main",
    "web",
    "--host",
    "127.0.0.1",
    "--port",
    String(port),
    "--no-browser",
  ];
}

function resolveServerLaunch(options) {
  const platform = options.platform;
  const repoRoot = options.repoRoot;
  const port = options.port;
  const sourceEnv = options.env || {};
  const runtimeEnv = {
    ...sourceEnv,
    QT_QPA_PLATFORM: "offscreen",
    PYVISTA_OFF_SCREEN: "true",
    PYTHONPATH: "src",
  };

  if (platform !== "win32") {
    return {
      command: "python3",
      args: serverArgs(port),
      options: { cwd: repoRoot, env: runtimeEnv },
      mode: "local",
    };
  }

  const mapped = uncToWslPath(repoRoot);
  const distro = sourceEnv.NAVIER_TWIN_WSL_DISTRO || mapped?.distro || DEFAULT_DISTRO;
  const projectRoot = sourceEnv.NAVIER_TWIN_WSL_PROJECT || mapped?.projectRoot;
  if (!projectRoot || !path.posix.isAbsolute(projectRoot)) {
    throw new Error(
      "Windows 실행에는 NAVIER_TWIN_WSL_PROJECT=/home/.../NavierTwin 설정이 필요합니다."
    );
  }

  return {
    command: "wsl.exe",
    args: [
      "-d",
      distro,
      "--cd",
      projectRoot,
      "--",
      "env",
      "PYTHONPATH=src",
      "QT_QPA_PLATFORM=offscreen",
      "PYVISTA_OFF_SCREEN=true",
      "python3",
      ...serverArgs(port),
    ],
    options: {
      cwd: undefined,
      env: sourceEnv,
      windowsHide: true,
    },
    mode: "wsl",
  };
}

module.exports = {
  resolveServerLaunch,
  serverArgs,
  uncToWslPath,
};
