#!/usr/bin/env bash
#
# Installer for camie-immich-tagger on Debian/Ubuntu systems such as the Immich LXC.
#
#   sudo ./install.sh                 # detect hardware, install everything
#   sudo ./install.sh --device intel  # force a specific accelerator
#   sudo ./install.sh --skip-model    # do not download the model
#
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv"
MODEL_DIR="$PROJECT_DIR/models"
HF_BASE="https://huggingface.co/Camais03/camie-tagger-v2/resolve/main"
MODEL_FILES=("camie-tagger-v2.onnx" "camie-tagger-v2-metadata.json")

DEVICE=""
SKIP_APT=0
SKIP_MODEL=0

info() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mWARNING:\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

usage() {
    sed -n '3,9p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --device)     DEVICE="${2:-}"; shift 2 ;;
        --device=*)   DEVICE="${1#*=}"; shift ;;
        --skip-apt)   SKIP_APT=1; shift ;;
        --skip-model) SKIP_MODEL=1; shift ;;
        -h|--help)    usage ;;
        *)            die "Unknown option: $1 (try --help)" ;;
    esac
done

if [ "$(id -u)" -eq 0 ]; then SUDO=""; else SUDO="sudo"; fi

# ---------------------------------------------------------------- hardware ---
detect_vendor() {
    local vendors="" id
    for f in /sys/class/drm/card*/device/vendor; do
        [ -r "$f" ] || continue
        id="$(tr -d '[:space:]' < "$f")"
        case "$id" in
            0x10de) vendors="$vendors nvidia" ;;
            0x8086) vendors="$vendors intel" ;;
            0x1002) vendors="$vendors amd" ;;
        esac
    done
    # Preference order when several GPUs are present.
    for want in nvidia intel amd; do
        case "$vendors" in *"$want"*) echo "$want"; return ;; esac
    done
    echo cpu
}

if [ -z "$DEVICE" ]; then
    DEVICE="$(detect_vendor)"
    info "Detected accelerator: $DEVICE"
else
    info "Accelerator forced to: $DEVICE"
fi

case "$DEVICE" in
    nvidia) ORT_PACKAGE="onnxruntime-gpu" ;;
    intel)  ORT_PACKAGE="onnxruntime-openvino" ;;
    amd)
        ORT_PACKAGE="onnxruntime"
        warn "ONNX Runtime publishes no AMD GPU wheel on PyPI. Installing the CPU build."
        warn "AMD integrated GPUs are not supported; discrete cards need a ROCm build from source."
        ;;
    cpu)    ORT_PACKAGE="onnxruntime"; warn "No GPU detected. Tagging will be slow." ;;
    *)      die "Unknown device '$DEVICE' (expected nvidia, intel, amd or cpu)" ;;
esac

# ---------------------------------------------------------- system packages ---
if [ "$SKIP_APT" -eq 0 ]; then
    info "Installing system packages"
    APT_PACKAGES="python3-venv python3-pip libimage-exiftool-perl curl"
    if [ "$DEVICE" = "intel" ]; then
        APT_PACKAGES="$APT_PACKAGES intel-opencl-icd"
    fi
    $SUDO apt-get update
    # shellcheck disable=SC2086
    $SUDO apt-get install -y $APT_PACKAGES
else
    info "Skipping system packages"
fi

command -v exiftool >/dev/null 2>&1 || die "exiftool is not on PATH after installation."

# ------------------------------------------------------------ virtualenv -----
info "Creating virtual environment at $VENV_DIR"
python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --quiet --upgrade pip setuptools wheel

info "Installing $ORT_PACKAGE"
"$VENV_DIR/bin/python" -m pip install --quiet "$ORT_PACKAGE"

info "Installing camie-immich-tagger"
"$VENV_DIR/bin/python" -m pip install --quiet -e "$PROJECT_DIR"

# ---------------------------------------------------------------- model ------
if [ "$SKIP_MODEL" -eq 0 ]; then
    mkdir -p "$MODEL_DIR"
    for file in "${MODEL_FILES[@]}"; do
        target="$MODEL_DIR/$file"
        if [ -s "$target" ]; then
            info "Model file already present: $file"
            continue
        fi
        info "Downloading $file (this takes a while for the 789 MB model)"
        curl -fL --progress-bar --continue-at - -o "$target" "$HF_BASE/$file?download=true" \
            || die "Download failed for $file"
    done
else
    info "Skipping model download"
fi

# ---------------------------------------------------------------- config -----
if [ ! -f "$PROJECT_DIR/.env" ]; then
    info "Creating .env from .env.example"
    cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
fi
chmod 600 "$PROJECT_DIR/.env"

# GPU access inside a container requires membership in the render/video groups.
if [ "$DEVICE" = "intel" ] || [ "$DEVICE" = "amd" ]; then
    if [ ! -e /dev/dri/renderD128 ]; then
        warn "/dev/dri/renderD128 is missing. Pass the render node into the LXC:"
        warn "  lxc.cgroup2.devices.allow: c 226:* rwm"
        warn "  lxc.mount.entry: /dev/dri dev/dri none bind,optional,create=dir"
    fi
fi

cat <<EOF

Installation finished.

Next steps:
  1. Edit the configuration:   nano $PROJECT_DIR/.env
     Set CAMIE_SCAN_DIRS, IMMICH_URL, IMMICH_API_KEY and IMMICH_LIBRARY_IDS.
     Look up the library UUID with:
       curl -H "x-api-key: YOUR_KEY" "YOUR_IMMICH_URL/api/libraries"
  2. Check the installation:   $VENV_DIR/bin/camie-tagger doctor
  3. Try a small test run:     $VENV_DIR/bin/camie-tagger run --mode test --limit 5 --dry-run
  4. Tag the whole library:    $VENV_DIR/bin/camie-tagger run --mode all --immich-scan

EOF
