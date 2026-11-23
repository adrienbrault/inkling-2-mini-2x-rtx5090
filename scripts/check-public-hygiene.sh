#!/usr/bin/env bash
# Fails when the staged diff (default) or the whole checkout (--tree) contains something that must not reach the
# public repo: private or Tailscale addresses, local machine paths, assistant scratchpad paths, session identifiers,
# credential-shaped strings. CLAUDE.md lists the rules. Run before every commit; nothing can be unpublished afterwards.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
SELF=scripts/check-public-hygiene.sh
PATS=(
  '(^|[^0-9.])10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}'
  '(^|[^0-9.])192\.168\.[0-9]{1,3}\.[0-9]{1,3}'
  '(^|[^0-9.])172\.(1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3}'
  '(^|[^0-9.])100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.[0-9]{1,3}\.[0-9]{1,3}'
  '/Users/[A-Za-z]'
  '/home/[a-z]'
  '/private/tmp/'
  'claude-501'
  '/scratchpad/'
  'claude\.ai/code/session_'
  '(^|[^A-Za-z0-9])ghp_[A-Za-z0-9]{30,}'
  '(^|[^A-Za-z0-9])github_pat_[A-Za-z0-9_]{20,}'
  '(^|[^A-Za-z0-9])gho_[A-Za-z0-9]{30,}'
  '(^|[^A-Za-z0-9])hf_[A-Za-z0-9]{30,}'
  '(^|[^A-Za-z0-9])sk-[A-Za-z0-9_-]{30,}'
  'AKIA[0-9A-Z]{16}'
  'BEGIN [A-Z ]*PRIVATE KEY'
  'xox[bpa]-[A-Za-z0-9-]{20,}'
)
RE=$(IFS='|'; echo "${PATS[*]}")
# These known measurement-host references are permitted only inside the raw archive.
ARCHIVE_ONLY='(^|[^A-Za-z0-9_-])flan([^A-Za-z0-9_-]|$)|/srv/qwen5090|opera/dsv41'
# Exempt: the docker bridge gateway (172.17.0.1 is the same on every host), and lines carrying a hygiene-ok marker
# with a reason (fictional addresses inside frozen benchmark prompts).
EXEMPT='172\.17\.0\.1|hygiene-ok'
hits=0
if [ "${1:-}" = "--tree" ]; then
  while IFS= read -r -d '' f; do
    [[ $f == "$SELF" || ! -f $f ]] && continue
    pattern=$RE
    [[ $f == bench/results/* ]] || pattern="$pattern|$ARCHIVE_ONLY"
    grep -InE "$pattern" -- "$f" | grep -vE "$EXEMPT" | sed "s|^|$f:|" | grep . && hits=1
  done < <(git ls-files --cached --others --exclude-standard -z)
else
  # Match added lines before printing their path prefix.
  out=$(git diff --cached -U0 --diff-filter=AM -- . ":!$SELF" \
    | RE_GENERAL="$RE" RE_ARCHIVE="$ARCHIVE_ONLY" RE_EXEMPT="$EXEMPT" awk '
      BEGIN { general=ENVIRON["RE_GENERAL"]; archive_only=ENVIRON["RE_ARCHIVE"]; exempt=ENVIRON["RE_EXEMPT"] }
      /^\+\+\+ b\// { f=substr($0,7); next }
      /^\+/ && !/^\+\+\+/ {
        line=substr($0,2)
        if (line !~ exempt && (line ~ general || (f !~ /^bench\/results\// && line ~ archive_only))) print f ": " line
      }')
  [ -n "$out" ] && { echo "$out"; hits=1; }
fi
if [ "$hits" = 1 ]; then echo "check-public-hygiene: FAIL (see lines above)"; exit 1; fi
echo "check-public-hygiene: OK"
