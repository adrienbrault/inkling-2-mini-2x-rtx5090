# CLAUDE.md

Working agreement for this repository, for people and coding agents alike (`AGENTS.md` is a link to it).

This repository is meant to become public. It mirrors the serving configuration kept in a private infrastructure repo.
Its GitHub repository (github.com/adrienbrault/inkling-2-mini-2x-rtx5090) is private until the owner makes it
public; push committed work there. Everything committed must already be fit to
publish, because history is published as it is.

## Never commit

- Credentials in any form: tokens, registry logins, API keys, SSH material, resolved secret-manager values.
- Internal addresses: private (RFC1918) and Tailscale IPs, LAN hostnames, MAC addresses.
- Paths of the author's machines: home directories, temp and assistant scratchpad directories, session identifiers.
  The measurement host's documented layout under `/srv/qwen5090` (and its `flan` alias) may appear in raw results.
- Numbers that were not measured. A projection or an estimate does not go into a results table.

Run `scripts/check-public-hygiene.sh` before every commit. It checks the staged diff; `--tree` scans the checkout.
Stage explicit paths; `git add -A` and `git add .` are forbidden.

## README

The top half describes the served configuration only (`scripts/serve.sh` and the image from `docker/`), with numbers
measured on it, each dated and linked to its raw results under `bench/results/`. Comparisons with other
configurations go in `docs/CONFIGURATION.md`. Decode: per-stream rate and the aggregate (sum of concurrent streams'
rates); time to the first token is separate; a benchmark's wall-clock output tok/s is labelled as such.
