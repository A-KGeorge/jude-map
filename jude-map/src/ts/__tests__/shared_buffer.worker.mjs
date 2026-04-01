// Worker bootstrap: registers tsx ESM hooks before loading the TypeScript entry.
import { register } from "tsx/esm/api";
import { workerData } from "node:worker_threads";

const DEBUG = process.env.JUDE_DEBUG === "1";
if (DEBUG)
  process.stderr.write(
    `[jude-map worker.mjs] starting, command=${workerData?.command}\n`,
  );

const unregister = register();

const { href } = new URL("./shared_buffer.test.ts", import.meta.url);
if (DEBUG) process.stderr.write(`[jude-map worker.mjs] importing ${href}\n`);

await import(href);

if (DEBUG)
  process.stderr.write(
    `[jude-map worker.mjs] import done, unregistering tsx\n`,
  );
unregister();

if (DEBUG) process.stderr.write(`[jude-map worker.mjs] done\n`);
