#!/usr/bin/env bash
# Shared model and client catalogue for every launcher and start script.
#
# One place decides which installs exist, what they are called and where
# they live, so the server launcher, the CLI launcher and the eight
# per-model start scripts cannot drift apart. A launcher that resolved the
# model itself and a start script that hardcoded a port is exactly how the
# "-fast" alias and the width-suffixed model id got out of step before.
#
# The client list lives here for the same reason: the launcher launches them
# and the coder benchmark measures them, so both read one list and a test
# (benchmark/test_coder_clients.py) fails if either drifts from it.
#
# Every model serves on one port. The server switches models on demand by
# name, one resident at a time, so a port per install only left client
# configs to keep in step. TINYTITAN_PORT still overrides it: the memory
# harness runs its own server on 8096 that way.
TINYTITAN_DEFAULT_PORT=8080

# --- Clients ---------------------------------------------------------------
#
# Every client this checkout launches or measures, in menu order:
#
#   id|label|kind|binaries
#
# `coder` clients are the ones a benchmark can ask a question and score; each
# has a command builder in benchmark/coder_cli_benchmark.py. `editor` clients
# are configured and opened instead: Zed's CLI has no non-interactive prompt
# mode (it opens windows, diffs files and pipes stdin), so it can be wired and
# checked, never prompted.
TINYTITAN_CLIENTS=(
  "codex|Codex|coder|codex"
  "claude|Claude Code|coder|claude"
  "qwen|Qwen Code|coder|qwen qwen-code"
  "opencode|OpenCode|coder|opencode"
  "zed|Zed editor|editor|zed"
)

# tinytitan_client_ids -> one client id per line, in menu order.
tinytitan_client_ids() {
  local entry
  for entry in "${TINYTITAN_CLIENTS[@]}"; do printf '%s\n' "${entry%%|*}"; done
}

# tinytitan_client_field <id> <2 label | 3 kind | 4 binaries> -> that field.
# Returns 1 for an id the list does not carry.
tinytitan_client_field() {
  local entry rest
  for entry in "${TINYTITAN_CLIENTS[@]}"; do
    [[ "${entry%%|*}" == "$1" ]] || continue
    rest="${entry#*|}"
    case "$2" in
      2) printf '%s\n' "${rest%%|*}" ;;
      3) rest="${rest#*|}"; printf '%s\n' "${rest%%|*}" ;;
      4) printf '%s\n' "${rest#*|}" ;;
      *) return 1 ;;
    esac
    return 0
  done
  return 1
}

tinytitan_client_label()    { tinytitan_client_field "$1" 2; }
tinytitan_client_kind()     { tinytitan_client_field "$1" 3; }
tinytitan_client_binaries() { tinytitan_client_field "$1" 4; }

# tinytitan_client_ids_csv <separator> -> "codex<sep>claude<sep>…", for help text.
tinytitan_client_ids_csv() {
  local separator="${1:-|}" first=1 id
  while IFS= read -r id; do
    (( first )) || printf '%s' "$separator"
    printf '%s' "$id"
    first=0
  done < <(tinytitan_client_ids)
  printf '\n'
}

# tinytitan_coder_client_ids -> just the clients a benchmark can prompt.
tinytitan_coder_client_ids() {
  local id
  while IFS= read -r id; do
    [[ "$(tinytitan_client_kind "$id")" == "coder" ]] && printf '%s\n' "$id"
  done < <(tinytitan_client_ids)
}

# tinytitan_resolve_model <key> -> TINYTITAN_MODEL_{KEY,STEM,LABEL,ENGINES,THINKING,FAMILY}
#
# ENGINES, THINKING and FAMILY are what the built-in fallback list needs to
# describe an install truthfully when the server cannot report its catalog:
# which engines can serve the family, default first, the thinking levels its
# chat template renders, and the family's own name as the runtime spells it
# (the engine message names it). The catalog is still the better source -- it
# reads all three off the install -- and the fallback is only reached without
# it.
tinytitan_resolve_model() {
  case "${1:-}" in
    ornith|ornith15|ornith1.5)
      TINYTITAN_MODEL_KEY=ornith
      TINYTITAN_MODEL_STEM="ornith-1.5_35B_A3B"
      TINYTITAN_MODEL_LABEL="Ornith 1.5 35B-A3B"
      TINYTITAN_MODEL_FAMILY=qwen36
      TINYTITAN_MODEL_ENGINES=gpu
      TINYTITAN_MODEL_THINKING="off,on" ;;
    qwen36|qwen3.6)
      TINYTITAN_MODEL_KEY=qwen36
      TINYTITAN_MODEL_STEM="qwen3.6_35B_A3B"
      TINYTITAN_MODEL_LABEL="Qwen 3.6 35B-A3B"
      TINYTITAN_MODEL_FAMILY=qwen36
      TINYTITAN_MODEL_ENGINES=gpu
      TINYTITAN_MODEL_THINKING="off,on" ;;
    agentworld|aw)
      TINYTITAN_MODEL_KEY=agentworld
      TINYTITAN_MODEL_STEM="qwen-agentworld_35B_A3B"
      TINYTITAN_MODEL_LABEL="Qwen-AgentWorld 35B-A3B"
      TINYTITAN_MODEL_FAMILY=qwen36
      TINYTITAN_MODEL_ENGINES=gpu
      TINYTITAN_MODEL_THINKING="off,on" ;;
    katcoder|kat|kat-coder)
      TINYTITAN_MODEL_KEY=katcoder
      TINYTITAN_MODEL_STEM="kat-coder-v2.5_35B_A3B"
      TINYTITAN_MODEL_LABEL="KAT-Coder-V2.5-Dev 35B-A3B"
      TINYTITAN_MODEL_FAMILY=qwen36
      TINYTITAN_MODEL_ENGINES=gpu
      TINYTITAN_MODEL_THINKING="off,on" ;;
    qwen38|qwen3.8)
      TINYTITAN_MODEL_KEY=qwen38
      TINYTITAN_MODEL_STEM="qwen3.8-flash-next_125B_A6B"
      TINYTITAN_MODEL_LABEL="Qwen3.8-Flash-Next 125B-A6B"
      TINYTITAN_MODEL_FAMILY=qwen38flash
      TINYTITAN_MODEL_ENGINES=gpu
      TINYTITAN_MODEL_THINKING="off,low,medium,xhigh" ;;
    # The dense models: the one shape both engines implement, GPU by default.
    qwen35-2b|qwen3.5-2b)
      TINYTITAN_MODEL_KEY=qwen35-2b
      TINYTITAN_MODEL_STEM="qwen3.5_2B"
      TINYTITAN_MODEL_LABEL="Qwen 3.5 2B"
      TINYTITAN_MODEL_FAMILY=qwen3_5_dense
      TINYTITAN_MODEL_ENGINES="gpu,cpu"
      TINYTITAN_MODEL_THINKING="off,on" ;;
    qwen35-4b|qwen3.5-4b)
      TINYTITAN_MODEL_KEY=qwen35-4b
      TINYTITAN_MODEL_STEM="qwen3.5_4B"
      TINYTITAN_MODEL_LABEL="Qwen 3.5 4B"
      TINYTITAN_MODEL_FAMILY=qwen3_5_dense
      TINYTITAN_MODEL_ENGINES="gpu,cpu"
      TINYTITAN_MODEL_THINKING="off,on" ;;
    qwen35-9b|qwen3.5-9b)
      TINYTITAN_MODEL_KEY=qwen35-9b
      TINYTITAN_MODEL_STEM="qwen3.5_9B"
      TINYTITAN_MODEL_LABEL="Qwen 3.5 9B"
      TINYTITAN_MODEL_FAMILY=qwen3_5_dense
      TINYTITAN_MODEL_ENGINES="gpu,cpu"
      TINYTITAN_MODEL_THINKING="off,on" ;;
    *)
      echo "unknown AI model: ${1:-} (ornith|qwen36|agentworld|katcoder|qwen38|qwen35-2b|qwen35-4b|qwen35-9b)" >&2
      return 2 ;;
  esac
}

# tinytitan_resolve_quant <4|8|4bit|8bit> -> TINYTITAN_QUANT ("4bit"/"8bit"), TINYTITAN_QUANT_DIR ("4Bit"/"8Bit")
tinytitan_resolve_quant() {
  case "${1:-}" in
    4|4bit) TINYTITAN_QUANT=4bit; TINYTITAN_QUANT_DIR=4Bit ;;
    8|8bit) TINYTITAN_QUANT=8bit; TINYTITAN_QUANT_DIR=8Bit ;;
    *) echo "unknown quantization: ${1:-} (4|8)" >&2; return 2 ;;
  esac
}

# tinytitan_model_port -> the one port every model is served on. Kept as a
# function because callers and notes still ask for it by name.
tinytitan_model_port() {
  echo "$TINYTITAN_DEFAULT_PORT"
}

# Every model this checkout knows about, for help text and for the fallback
# list when the server cannot report its catalog. One list: the catalog is
# still the source of what is installed.
TINYTITAN_ALL_MODELS=(ornith qwen36 agentworld katcoder qwen38 qwen35-2b qwen35-4b qwen35-9b)

# --- Installed models, from the server's catalog -----------------------
#
# The server knows what is installed -- GPU installs and CPU snapshots,
# their ids, widths and the thinking levels each one supports -- so the
# launcher asks it instead of keeping a second list that goes stale:
# `TinyTitanServer --catalog --models-dir <dir>` prints JSON. TINYTITAN_CATALOG_JSON
# reads the same JSON from a file instead, so the launcher can be exercised
# against tools/testdata/catalog-example.json without a build.
#
# tinytitan_load_catalog <binary> <models-dir> fills parallel arrays, one entry
# per model and quantization, GPU entries first:
#   TINYTITAN_CAT_ID  TINYTITAN_CAT_NAME  TINYTITAN_CAT_QUANT (4|8)  TINYTITAN_CAT_BACKEND (gpu|cpu)
#   TINYTITAN_CAT_PATH  TINYTITAN_CAT_THINKING (comma-separated, off first)  TINYTITAN_CAT_SIZE (GB, or -)
#   TINYTITAN_CAT_FAMILY (the manifest's family, e.g. qwen3_5_dense)
#   TINYTITAN_CAT_ENGINES (comma-separated engines that can serve it, default first)
# and returns 1 with the reason in TINYTITAN_CATALOG_ERROR when there is no
# usable catalog.

# Levels in the order they are offered; anything newer the server adds is
# kept, after these.
TINYTITAN_CATALOG_PARSER='
import json, sys
LEVELS = ["off", "on", "minimal", "low", "medium", "high", "xhigh", "max"]
def field(value):
    text = str(value)
    if not text or any(c in text for c in "\t\r\n"):
        raise ValueError("empty or multi-line field")
    return text
try:
    models = json.load(sys.stdin)["models"]
except (ValueError, KeyError, TypeError) as error:
    sys.exit("catalog: not the expected JSON (%s)" % error)
if not isinstance(models, list):
    sys.exit("catalog: \"models\" is not a list")
gpu, cpu = [], []
for model in models:
    try:
        backend = field(model["backend"])
        if backend not in ("gpu", "cpu"):
            raise ValueError("backend %r" % backend)
        listed = [str(level) for level in (model.get("thinking") or ["off"])]
        levels = [l for l in LEVELS if l in listed] + [l for l in listed if l not in LEVELS]
        size = model.get("size_gb")
        family = field(model.get("family") or "-")
        engines = field(model.get("engines") or backend)
        row = [field(model["id"]), field(model["name"]), field(int(model["quant"])),
               backend, field(model["path"]), field(",".join(levels)),
               "%.1f" % float(size) if size is not None else "-", family, engines]
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        name = model.get("id") if isinstance(model, dict) else model
        print("catalog: skipping %r (%s)" % (name, error), file=sys.stderr)
        continue
    (gpu if backend == "gpu" else cpu).append(row)
for row in gpu + cpu:
    print("\t".join(row))
'

tinytitan_reset_catalog() {
  TINYTITAN_CAT_ID=(); TINYTITAN_CAT_NAME=(); TINYTITAN_CAT_QUANT=(); TINYTITAN_CAT_BACKEND=()
  TINYTITAN_CAT_PATH=(); TINYTITAN_CAT_THINKING=(); TINYTITAN_CAT_SIZE=(); TINYTITAN_CAT_FAMILY=()
  TINYTITAN_CAT_ENGINES=(); TINYTITAN_CATALOG_MISSING=()
}

tinytitan_load_catalog() {
  local binary="$1" models_dir="$2" json rows
  local id name quant backend path thinking size family engines
  TINYTITAN_CATALOG_ERROR=""
  tinytitan_reset_catalog
  if [[ -n "${TINYTITAN_CATALOG_JSON:-}" ]]; then
    if ! json="$(cat "$TINYTITAN_CATALOG_JSON" 2>/dev/null)"; then
      TINYTITAN_CATALOG_ERROR="cannot read TINYTITAN_CATALOG_JSON=$TINYTITAN_CATALOG_JSON"
      return 1
    fi
  elif [[ ! -x "$binary" ]]; then
    TINYTITAN_CATALOG_ERROR="no server binary at $binary"
    return 1
  elif ! json="$("$binary" --catalog --models-dir "$models_dir" 2>/dev/null)"; then
    TINYTITAN_CATALOG_ERROR="$(basename "$binary") --catalog failed; this build may predate it"
    return 1
  fi
  if ! command -v python3 >/dev/null 2>&1; then
    TINYTITAN_CATALOG_ERROR="python3 is needed to read it"
    return 1
  fi
  if ! rows="$(printf '%s' "$json" | python3 -c "$TINYTITAN_CATALOG_PARSER")"; then
    TINYTITAN_CATALOG_ERROR="its output could not be read"
    return 1
  fi
  # Tab-separated with no empty fields (the parser writes "-"), because
  # read collapses runs of tabs and an empty column would shift the rest.
  while IFS=$'\t' read -r id name quant backend path thinking size family engines; do
    [[ -n "$id" ]] || continue
    TINYTITAN_CAT_ID+=("$id"); TINYTITAN_CAT_NAME+=("$name"); TINYTITAN_CAT_QUANT+=("$quant")
    TINYTITAN_CAT_BACKEND+=("$backend"); TINYTITAN_CAT_PATH+=("$path")
    TINYTITAN_CAT_THINKING+=("$thinking"); TINYTITAN_CAT_SIZE+=("$size")
    TINYTITAN_CAT_FAMILY+=("$family"); TINYTITAN_CAT_ENGINES+=("$engines")
  done <<< "$rows"
  if (( ${#TINYTITAN_CAT_ID[@]} == 0 )); then
    TINYTITAN_CATALOG_ERROR="it lists no installed models"
    return 1
  fi
}

# tinytitan_catalog_keep_installed: drop every entry whose install directory is
# gone, so what the launcher presents is what the disk holds whatever the
# catalog came from. TINYTITAN_CATALOG_JSON can name an older build's output and a
# model can be deleted between two runs; neither may leave a model or a width
# on the menu that cannot be loaded. Returns 1 when nothing is left.
tinytitan_catalog_keep_installed() {
  local i
  local ids=() names=() quants=() backends=() paths=()
  local thinking=() sizes=() families=() engines=()
  for (( i = 0; i < ${#TINYTITAN_CAT_ID[@]}; i++ )); do
    [[ -e "${TINYTITAN_CAT_PATH[$i]}" ]] || continue
    ids+=("${TINYTITAN_CAT_ID[$i]}"); names+=("${TINYTITAN_CAT_NAME[$i]}")
    quants+=("${TINYTITAN_CAT_QUANT[$i]}"); backends+=("${TINYTITAN_CAT_BACKEND[$i]}")
    paths+=("${TINYTITAN_CAT_PATH[$i]}"); thinking+=("${TINYTITAN_CAT_THINKING[$i]}")
    sizes+=("${TINYTITAN_CAT_SIZE[$i]}"); families+=("${TINYTITAN_CAT_FAMILY[$i]}")
    engines+=("${TINYTITAN_CAT_ENGINES[$i]}")
  done
  tinytitan_reset_catalog
  (( ${#ids[@]} > 0 )) || return 1
  TINYTITAN_CAT_ID=("${ids[@]}"); TINYTITAN_CAT_NAME=("${names[@]}")
  TINYTITAN_CAT_QUANT=("${quants[@]}"); TINYTITAN_CAT_BACKEND=("${backends[@]}")
  TINYTITAN_CAT_PATH=("${paths[@]}"); TINYTITAN_CAT_THINKING=("${thinking[@]}")
  TINYTITAN_CAT_SIZE=("${sizes[@]}"); TINYTITAN_CAT_FAMILY=("${families[@]}")
  TINYTITAN_CAT_ENGINES=("${engines[@]}")
}

# tinytitan_static_catalog <models-dir>: the same arrays from the built-in list,
# for a server that cannot report its catalog. No ids: such a server reports
# its one id once up.
#
# Only installs that are on disk are listed: this menu is what the person is
# about to load, so a model or a width that is not in models/ is not offered.
# The labels left out are collected in TINYTITAN_CATALOG_MISSING, so the caller can
# say in one line what TinyTitan supports but this checkout does not have. Returns
# 1 with TINYTITAN_CATALOG_ERROR set when models/ holds none of them.
tinytitan_static_catalog() {
  local models_dir="$1" key bits dir added
  local missing=()
  tinytitan_reset_catalog
  for key in "${TINYTITAN_ALL_MODELS[@]}"; do
    tinytitan_resolve_model "$key"
    added=0
    for bits in 8 4; do
      tinytitan_resolve_quant "$bits"
      dir="$models_dir/${TINYTITAN_MODEL_STEM}_${TINYTITAN_QUANT_DIR}"
      [[ -e "$dir" ]] || continue
      TINYTITAN_CAT_ID+=("-"); TINYTITAN_CAT_NAME+=("$TINYTITAN_MODEL_LABEL"); TINYTITAN_CAT_QUANT+=("$bits")
      TINYTITAN_CAT_BACKEND+=("${TINYTITAN_MODEL_ENGINES%%,*}")
      TINYTITAN_CAT_PATH+=("$dir"); TINYTITAN_CAT_THINKING+=("$TINYTITAN_MODEL_THINKING")
      TINYTITAN_CAT_SIZE+=("-"); TINYTITAN_CAT_FAMILY+=("$TINYTITAN_MODEL_FAMILY")
      TINYTITAN_CAT_ENGINES+=("$TINYTITAN_MODEL_ENGINES")
      added=1
    done
    (( added )) || missing+=("$TINYTITAN_MODEL_LABEL")
  done
  (( ${#missing[@]} > 0 )) && TINYTITAN_CATALOG_MISSING=("${missing[@]}")
  if (( ${#TINYTITAN_CAT_ID[@]} == 0 )); then
    TINYTITAN_CATALOG_ERROR="no install under $models_dir matches the built-in list"
    return 1
  fi
}

# tinytitan_catalog_find_id <id> -> echoes the index of the entry with that id.
tinytitan_catalog_find_id() {
  local i
  for (( i = 0; i < ${#TINYTITAN_CAT_ID[@]}; i++ )); do
    if [[ "${TINYTITAN_CAT_ID[$i]}" != "-" && "${TINYTITAN_CAT_ID[$i]}" == "$1" ]]; then
      echo "$i"
      return 0
    fi
  done
  return 1
}

# tinytitan_catalog_find_dir <install directory name> -> echoes the index of the
# entry installed there. This is how an install key and a width (ornith 8)
# find their entry: the directory name is fixed by the installer, while
# the id is the server's to choose.
tinytitan_catalog_find_dir() {
  local i
  for (( i = 0; i < ${#TINYTITAN_CAT_PATH[@]}; i++ )); do
    if [[ "$(basename "${TINYTITAN_CAT_PATH[$i]}")" == "$1" ]]; then
      echo "$i"
      return 0
    fi
  done
  return 1
}

# --- Persistent memory -------------------------------------------------
#
# Off unless TINYTITAN_MEMORY=1. Memory runs inside the server process, so there
# is no database to install or start, and it takes what it needs: measured,
# a hundred-chapter novel over ten sessions was about 100 KB. There is no
# RAM ceiling by default; TINYTITAN_MEMORY_CACHE_MIB sets one for anyone who
# wants it. The files live in a dedicated folder under the checkout,
# <TinyTitan>/memory, beside models/ (override with TINYTITAN_MEMORY_DIR). One file
# per project; a project untouched for 30 days is deleted, and at most 100
# are kept (TINYTITAN_MEMORY_RETENTION_DAYS, TINYTITAN_MEMORY_MAX_WORKSPACES).
#
# The workspace defaults to the directory the launcher was run from, which
# is the repository being worked on, so two checkouts never share memory.

tinytitan_export_memory_environment() {
  local workspace_dir="${1:-$PWD}"
  export TINYTITAN_MEMORY="${TINYTITAN_MEMORY:-0}"
  [[ "$TINYTITAN_MEMORY" == "1" ]] || return 0
  # The home directory, its parent and the root are not projects. A server
  # launched from one and used for everything would put a novel and a
  # codebase in one fact store, so this refuses to start rather than mix.
  # The server applies the same rule on its own; this just says it earlier
  # and in the terminal the person is looking at.
  if [[ -z "${TINYTITAN_MEMORY_WORKSPACE:-}" ]]; then
    local resolved home_dir
    resolved="$(cd "$workspace_dir" 2>/dev/null && pwd -P)"
    home_dir="$(cd "$HOME" 2>/dev/null && pwd -P)"
    if [[ "$resolved" == "$home_dir" || "$resolved" == "$(dirname "$home_dir")" || "$resolved" == "/" ]]; then
      echo "ERROR: TINYTITAN_MEMORY=1 but this is launched from $resolved, which is not a project directory." >&2
      echo "       Memory would collect every project into one store. Either:" >&2
      echo "         cd <your project>   and run the start script again" >&2
      echo "       or name the workspace explicitly:" >&2
      echo "         TINYTITAN_MEMORY_WORKSPACE=my-project <start script>" >&2
      exit 2
    fi
  fi
  export TINYTITAN_WORKSPACE_DIR="${TINYTITAN_WORKSPACE_DIR:-$workspace_dir}"
  export TINYTITAN_MEMORY_NAMESPACE="${TINYTITAN_MEMORY_NAMESPACE:-tinytitan}"
  local tinytitan_root
  tinytitan_root="${BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
  export TINYTITAN_MEMORY_DIR="${TINYTITAN_MEMORY_DIR:-$tinytitan_root/memory}"
}
