// Must be the FIRST import in index.ts. ESM evaluates all of a module's
// imports (recursively, in source order) before running that module's own
// top-level statements — so a `dotenv.config()` call sitting in index.ts's
// body, even near the top of the file, still runs after every import above
// it has already been evaluated. Modules like opensymbols.ts read
// process.env at their own top level, so they'd see an empty environment
// unless dotenv is loaded from a module that is itself imported first.
import dotenv from 'dotenv';
import path from 'path';

dotenv.config({ path: path.resolve(process.cwd(), '.env') });
