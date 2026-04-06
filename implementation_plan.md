# Implementation Plan: Production-Grade Logging

Replace all `print()` statements with structured logging using the Python `logging` module to meet production readiness standards.

## User Review Required

> [!IMPORTANT]
> The format of the output in `inference.py` will change from plain text to structured log lines (e.g., `2026-04-03 14:45:00 [INFO] [inference.py] ...`). If you use automated tools to parse the "START/STEP/END" markers, they should be updated to handle the log prefix.

## Proposed Changes

### SupportDesk Environment Core
#### [NEW] [logging_config.py](file:///c:/Users/91630/Downloads/A-Chi/supportdesk_env/logging_config.py)
- Create a shared `get_logger(name)` utility.
- Set a default level of `INFO`, configurable via the `LOG_LEVEL` environment variable.
- Format: `%(asctime)s [%(levelname)s] [%(name)s] - %(message)s`.

### Agent & Utilities
#### [MODIFY] [inference.py](file:///c:/Users/91630/Downloads/A-Chi/inference.py)
- Import and initialize the logger.
- Update `_print_start`, `_print_step`, and `_print_end` to use `logger.info()`.
- Replace all other `print()` calls with appropriate levels (`info`, `error`, `warning`).

#### [MODIFY] [smoke_test.py](file:///c:/Users/91630/Downloads/A-Chi/smoke_test.py)
- Replace progress prints with `logger.info()`.
- Replace error prints with `logger.error()`.

#### [MODIFY] [validate.py](file:///c:/Users/91630/Downloads/A-Chi/scripts/validate.py)
- Replace informative prints with logging.
- Note: Keep the final JSON result as a `print()` if it's used for shell piping/automation, or transition to a structured log.

## Open Questions
1. **Output Level**: Should we default the log level to `DEBUG` for more verbosity during development, or keep it at `INFO`?
2. **Clean Logs**: Would you like the "OpenEnv" markers (`[START]`, `[STEP]`) to remain "clean" without the timestamp prefix for easier parsing?

## Verification Plan

### Automated Tests
- Run `pytest` to ensure logging integration doesn't break logic.
- Run `python inference.py --offline` to verify log appearance.

### Manual Verification
- Run `python smoke_test.py` and verify terminal output.
