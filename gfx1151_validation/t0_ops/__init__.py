"""T0 primitive-op tests — the agent fan-out gate. Each op is a small, deterministic,
framework-agnostic reference fn + a seeded input generator, tagged with the inventory
family it covers. Agents add coverage by appending to REGISTRY in ops.py."""
