"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const packageConfig = require("../package.json");

test("desktop package declares reproducible Linux and Windows targets", () => {
  assert.equal(packageConfig.build.appId, "org.naviertwin.desktop");
  assert.deepEqual(packageConfig.build.linux.target, ["AppImage"]);
  assert.deepEqual(packageConfig.build.win.target, ["nsis"]);
  assert.equal(packageConfig.build.asar, true);
  assert.match(packageConfig.scripts["dist:linux"], /AppImage/);
  assert.match(packageConfig.scripts["dist:win"], /nsis/);
});

test("package includes every server lifecycle module", () => {
  const files = new Set(packageConfig.build.files);
  for (const required of [
    "main.js",
    "preload.js",
    "server-launch.js",
    "server-process.js",
    "splash.html",
  ]) {
    assert.ok(files.has(required), `missing packaged file: ${required}`);
  }
});
