# Freeman Hive Mind 3.2.2

`3.2.2` is a `hive_mind` bugfix prerelease for Redis-backed horizontal scaling.

## What Changed

- Fixed `RedisLockBackend.unlock()` to use an atomic Lua compare-and-delete script for owner-checked unlocks.
- Preserved plain `DEL` for `force=True` and owner-unchecked unlocks.
- Added tests for:
  - correct owner unlock
  - wrong owner rejection through the script path
  - force unlock ignoring owner
- Documented the Redis unlock consistency boundary in the hive runtime deployment notes.

## Why It Matters

The previous Redis unlock path performed `GET` and `DEL` as separate commands. If a lease expired between those calls and another worker acquired the same node, the old owner could delete the new owner's lock. The Lua script makes the owner check and delete one Redis-side operation.

## Validation

- `pytest tests/test_hive_mind.py` -> `19 passed`
