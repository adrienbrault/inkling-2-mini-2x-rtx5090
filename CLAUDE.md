# CLAUDE.md

Working agreement for this repository, for people and coding agents alike (`AGENTS.md` is a link to it).

This repository documents the serving configuration for Inkling-2-Mini on two RTX 5090s. Write changes for
readers of the public checkout. Everything committed must be fit to publish, including the full history.

## Never commit

- Credentials in any form: tokens, registry logins, API keys, SSH material, resolved secret-manager values.
- Internal addresses: private (RFC1918) and Tailscale IPs, LAN hostnames, MAC addresses.
- Paths of the author's machines: home directories, temp and assistant scratchpad directories, session identifiers.
  The archived raw results may retain the measurement host's paths and alias. Use neutral names in public-facing prose.
- Numbers that were not measured. A projection or an estimate does not go into a results table.

Run `scripts/check-public-hygiene.sh` before every commit. It checks the staged diff; `--tree` scans the checkout.
Stage explicit paths; `git add -A` and `git add .` are forbidden.

## README

The top half describes the served configuration (`scripts/serve.sh` and the image from `docker/`) and identifies
the configuration behind each headline measurement. Date and link measurements to raw results under `bench/results/`. Comparisons with other
configurations go in `docs/CONFIGURATION.md`. Decode: per-stream rate and the aggregate (sum of concurrent streams'
rates); time to the first token is separate; a benchmark's wall-clock output tok/s is labelled as such.
