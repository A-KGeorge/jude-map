// Worker bootstrap: registers tsx ESM hooks before loading the TypeScript entry.
import { register } from "tsx/esm/api";
import { isMainThread, parentPort, workerData } from "node:worker_threads";

const unregister = register();

const { href } = new URL("./shared_buffer.test.ts", import.meta.url);
await import(href);

unregister();
