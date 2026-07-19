"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const {
  resolveServerLaunch,
  uncToWslPath,
} = require("../server-launch");

test("Linux launches the Python server directly", () => {
  const launch = resolveServerLaunch({
    platform: "linux",
    repoRoot: "/work/NavierTwin",
    port: 8890,
    env: { HOME: "/home/user" },
  });
  assert.equal(launch.command, "python3");
  assert.equal(launch.options.cwd, "/work/NavierTwin");
  assert.equal(launch.options.env.PYTHONPATH, "src");
  assert.ok(launch.args.includes("8890"));
});

test("WSL UNC roots map to distro and Linux path", () => {
  assert.deepEqual(
    uncToWslPath("\\\\wsl.localhost\\Ubuntu-24.04\\home\\user\\NavierTwin"),
    {
      distro: "Ubuntu-24.04",
      projectRoot: "/home/user/NavierTwin",
    }
  );
  assert.deepEqual(uncToWslPath("\\\\wsl$\\Ubuntu\\srv\\NavierTwin"), {
    distro: "Ubuntu",
    projectRoot: "/srv/NavierTwin",
  });
});

test("Windows launches through wsl.exe without a shell command", () => {
  const launch = resolveServerLaunch({
    platform: "win32",
    repoRoot: "\\\\wsl.localhost\\Ubuntu\\home\\user\\NavierTwin",
    port: 8877,
    env: {},
  });
  assert.equal(launch.command, "wsl.exe");
  assert.equal(launch.mode, "wsl");
  assert.deepEqual(launch.args.slice(0, 6), [
    "-d",
    "Ubuntu",
    "--cd",
    "/home/user/NavierTwin",
    "--",
    "env",
  ]);
  assert.ok(launch.args.includes("python3"));
  assert.equal(launch.args.includes("bash"), false);
  assert.equal(launch.args.includes("sh"), false);
});

test("explicit WSL settings support a Windows checkout", () => {
  const launch = resolveServerLaunch({
    platform: "win32",
    repoRoot: "C:\\NavierTwin",
    port: 8877,
    env: {
      NAVIER_TWIN_WSL_DISTRO: "Ubuntu-24.04",
      NAVIER_TWIN_WSL_PROJECT: "/opt/naviertwin",
    },
  });
  assert.deepEqual(launch.args.slice(0, 4), [
    "-d",
    "Ubuntu-24.04",
    "--cd",
    "/opt/naviertwin",
  ]);
});

test("Windows checkout without WSL project setting fails early", () => {
  assert.throws(
    () =>
      resolveServerLaunch({
        platform: "win32",
        repoRoot: "C:\\NavierTwin",
        port: 8877,
        env: {},
      }),
    /NAVIER_TWIN_WSL_PROJECT/
  );
});
