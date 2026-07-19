"use strict";

const assert = require("node:assert/strict");
const http = require("node:http");
const test = require("node:test");

const {
  checkHealth,
  findFreePort,
  stopChildProcess,
  waitForServer,
} = require("../server-process");

test("findFreePort skips a bound port", async () => {
  const server = http.createServer((_request, response) => response.end("ok"));
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const occupied = server.address().port;
  const free = await findFreePort(occupied, 3);
  assert.notEqual(free, occupied);
  await new Promise((resolve) => server.close(resolve));
});

test("health check requires HTTP 200", async () => {
  const server = http.createServer((request, response) => {
    response.statusCode = request.url === "/index.html" ? 200 : 404;
    response.end();
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const port = server.address().port;
  assert.equal(await checkHealth(port), true);
  assert.equal(await checkHealth(port, { path: "/missing" }), false);
  await new Promise((resolve) => server.close(resolve));
});

test("waitForServer polls until ready", async () => {
  let probes = 0;
  const ready = await waitForServer(0, 100, {
    intervalMs: 1,
    healthCheck: async () => {
      probes += 1;
      return probes === 3;
    },
  });
  assert.equal(ready, true);
  assert.equal(probes, 3);
});

test("stopChildProcess is scoped and idempotent", () => {
  const signals = [];
  const child = {
    killed: false,
    exitCode: null,
    kill(signal) {
      signals.push(signal);
      this.killed = true;
      return true;
    },
  };
  assert.equal(stopChildProcess(child, { timeoutMs: 5 }), true);
  assert.deepEqual(signals, ["SIGTERM"]);
  assert.equal(stopChildProcess(child), false);
});
