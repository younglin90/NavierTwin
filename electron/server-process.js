"use strict";

const http = require("http");
const net = require("net");

function isPortFree(port, host = "127.0.0.1") {
  return new Promise((resolve) => {
    const server = net.createServer();
    server.once("error", () => resolve(false));
    server.once("listening", () => server.close(() => resolve(true)));
    server.listen(port, host);
  });
}

async function findFreePort(start, count = 100) {
  for (let port = start; port < start + count; port += 1) {
    // Port probes are sequential to avoid binding races.
    // eslint-disable-next-line no-await-in-loop
    if (await isPortFree(port)) return port;
  }
  throw new Error("사용 가능한 포트를 찾지 못했습니다.");
}

function checkHealth(port, options = {}) {
  const host = options.host || "127.0.0.1";
  const pathname = options.path || "/index.html";
  const timeoutMs = options.timeoutMs || 800;
  return new Promise((resolve) => {
    const request = http.get(
      { host, port, path: pathname, timeout: timeoutMs },
      (response) => {
        response.resume();
        resolve(response.statusCode === 200);
      }
    );
    request.on("error", () => resolve(false));
    request.on("timeout", () => {
      request.destroy();
      resolve(false);
    });
  });
}

async function waitForServer(port, timeoutMs, options = {}) {
  const intervalMs = options.intervalMs || 500;
  const healthCheck = options.healthCheck || checkHealth;
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    // eslint-disable-next-line no-await-in-loop
    if (await healthCheck(port)) return true;
    // eslint-disable-next-line no-await-in-loop
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
  return false;
}

function stopChildProcess(child, options = {}) {
  const timeoutMs = options.timeoutMs || 2000;
  if (!child || child.killed || child.exitCode !== null) return false;
  try {
    child.kill("SIGTERM");
  } catch (_error) {
    return false;
  }
  const timer = setTimeout(() => {
    if (!child.killed && child.exitCode === null) {
      try {
        child.kill("SIGKILL");
      } catch (_error) {
        // Process already exited between state check and signal.
      }
    }
  }, timeoutMs);
  timer.unref();
  return true;
}

module.exports = {
  checkHealth,
  findFreePort,
  isPortFree,
  stopChildProcess,
  waitForServer,
};
