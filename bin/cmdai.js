#!/usr/bin/env node
const { spawn } = require("child_process");
const path = require("path");

const rootDir = path.resolve(__dirname, "..");
const scriptPath = path.join(rootDir, "cmdai.py");
const args = [scriptPath, ...process.argv.slice(2)];

const env = {
  ...process.env,
  CMDAI_ROOT: rootDir,
  PYTHONIOENCODING: "utf-8",
};

const child = spawn(process.platform === "win32" ? "python" : "python3", args, {
  stdio: "inherit",
  env,
});

child.on("exit", (code) => {
  process.exit(code == null ? 0 : code);
});
