#!/usr/bin/env bash
# Lambda Cloud helper for GPU training runs. See LAMBDA.md for the full flow.
#
# The API key lives OUTSIDE the repo in ~/.config/lambda/api_key (chmod 600).
# Never commit it. SSH uses ~/.ssh/id_ed25519 (registered as "linda-local").
#
#   scripts/lambda.sh status                 # running instances + IPs
#   scripts/lambda.sh types                  # instance types with capacity
#   scripts/lambda.sh launch [type] [region] # default: gpu_1x_a100_sxm4
#   scripts/lambda.sh setup  <ip>            # rsync repo + install GPU deps + validate
#   scripts/lambda.sh train  <ip> <env> [extra train_mjx.py flags...]
#   scripts/lambda.sh watch  <ip>            # tail the remote training log
#   scripts/lambda.sh sync   <ip>            # pull runs/ back to this machine
#   scripts/lambda.sh terminate <instance-id>

set -euo pipefail

KEY_FILE="$HOME/.config/lambda/api_key"
[ -f "$KEY_FILE" ] || { echo "missing $KEY_FILE (Lambda API key)"; exit 1; }
KEY=$(cat "$KEY_FILE")
API="https://cloud.lambdalabs.com/api/v1"
SSH_KEY_NAME="linda-local"
SSH_OPTS=(-i "$HOME/.ssh/id_ed25519" -o StrictHostKeyChecking=accept-new)
REMOTE_DIR="rl-mon"

api() { curl -s -u "$KEY:" "$@"; }

repo_root() { cd "$(dirname "$0")/.." && pwd; }

cmd_status() {
  api "$API/instances" | python3 -c '
import json, sys
data = json.load(sys.stdin)["data"]
if not data: print("no running instances"); raise SystemExit
for i in data:
    t = i["instance_type"]
    print(f"{i['\''id'\'']}  {i.get('\''name'\'') or '\''-'\'':24s} "
          f"{t['\''name'\'']:20s} ${t['\''price_cents_per_hour'\'']/100:.2f}/hr  "
          f"{i['\''region'\'']['\''name'\'']:12s} {i['\''status'\'']:10s} {i.get('\''ip'\'') or '\''-'\''}")'
}

cmd_types() {
  api "$API/instance-types" | python3 -c '
import json, sys
for name, info in sorted(json.load(sys.stdin)["data"].items()):
    regions = [r["name"] for r in info.get("regions_with_capacity_available", [])]
    if regions:
        t = info["instance_type"]
        print(f"{name:32s} ${t['\''price_cents_per_hour'\'']/100:6.2f}/hr  {regions}")'
}

cmd_launch() {
  local type="${1:-gpu_1x_a100_sxm4}"
  local region="${2:-}"
  if [ -z "$region" ]; then
    region=$(api "$API/instance-types" | python3 -c "
import json, sys
info = json.load(sys.stdin)['data']['$type']
regions = info.get('regions_with_capacity_available', [])
print(regions[0]['name'] if regions else '')")
    [ -n "$region" ] || { echo "no capacity for $type; try: scripts/lambda.sh types"; exit 1; }
  fi
  echo "launching $type in $region ..."
  local id
  id=$(api -X POST "$API/instance-operations/launch" \
    -H "Content-Type: application/json" \
    -d "{\"instance_type_name\":\"$type\",\"region_name\":\"$region\",\"ssh_key_names\":[\"$SSH_KEY_NAME\"],\"quantity\":1,\"name\":\"rl-mon-train\"}" \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['data']['instance_ids'][0]) if 'data' in d else sys.exit(json.dumps(d))")
  echo "instance id: $id — waiting for it to boot (this takes a few minutes) ..."
  while true; do
    sleep 20
    local out
    out=$(api "$API/instances/$id" | python3 -c \
      "import json,sys; i=json.load(sys.stdin)['data']; print(i['status'], i.get('ip') or '')")
    echo "  $out"
    case "$out" in active\ *) echo "ready: ${out#active }"; break;; esac
  done
}

cmd_setup() {
  local ip="$1"; local root; root=$(repo_root)
  echo "== rsync repo -> ubuntu@$ip:$REMOTE_DIR"
  rsync -az -e "ssh ${SSH_OPTS[*]}" \
    --exclude '.venv*' --exclude 'node_modules' --exclude 'web*/' \
    --exclude 'runs' --exclude 'logs' --exclude '.git' --exclude '__pycache__' \
    "$root/" "ubuntu@$ip:$REMOTE_DIR/"
  echo "== install GPU stack + verify CUDA + validate env ports"
  # uv provides a managed Python 3.12 (Lambda images ship 3.10, too old for
  # our pinned jax) and installs the requirements in seconds.
  ssh "${SSH_OPTS[@]}" "ubuntu@$ip" "
    set -e
    command -v ~/.local/bin/uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh -s -- -q
    cd $REMOTE_DIR
    ~/.local/bin/uv venv --python 3.12 venv --allow-existing -q
    ~/.local/bin/uv pip install -q -p venv -r requirements-mjx.txt
    ./venv/bin/python -c 'import jax; ds=jax.devices(); print(\"jax devices:\", ds); assert ds[0].platform.lower() in (\"gpu\",\"cuda\"), \"NO GPU VISIBLE\"'
    ./venv/bin/python -m mjx_envs.validate --steps 100
  "
  echo "setup OK"
}

cmd_train() {
  local ip="$1"; shift
  local env="$1"; shift
  local name="${env%-v*}"; name=$(echo "$name" | tr '[:upper:]' '[:lower:]')
  local run="${name}_mjx_$(date +%Y%m%d_%H%M%S)"
  echo "== starting $env as $run (survives SSH disconnects)"
  ssh "${SSH_OPTS[@]}" "ubuntu@$ip" "
    cd $REMOTE_DIR
    nohup ./venv/bin/python train_mjx.py --env $env --run-name $run $* \
      > train_$run.log 2>&1 &
    echo \$! > train_$run.pid
    echo started, log: $REMOTE_DIR/train_$run.log
  "
  echo "watch with: scripts/lambda.sh watch $ip"
}

cmd_watch() {
  local ip="$1"
  ssh "${SSH_OPTS[@]}" "ubuntu@$ip" "cd $REMOTE_DIR && tail -n 30 -f \$(ls -t train_*.log | head -1)"
}

cmd_sync() {
  local ip="$1"; local root; root=$(repo_root)
  echo "== pulling runs/ back"
  rsync -az -e "ssh ${SSH_OPTS[*]}" "ubuntu@$ip:$REMOTE_DIR/runs/" "$root/runs/"
  echo "synced. export with: python export_onnx.py --all-envs"
}

cmd_terminate() {
  local id="$1"
  api -X POST "$API/instance-operations/terminate" \
    -H "Content-Type: application/json" \
    -d "{\"instance_ids\":[\"$id\"]}" | python3 -m json.tool
}

case "${1:-}" in
  status) cmd_status ;;
  types) cmd_types ;;
  launch) shift; cmd_launch "$@" ;;
  setup) shift; cmd_setup "$@" ;;
  train) shift; cmd_train "$@" ;;
  watch) shift; cmd_watch "$@" ;;
  sync) shift; cmd_sync "$@" ;;
  terminate) shift; cmd_terminate "$@" ;;
  *) grep '^#   scripts/' "$0" | sed 's/^# *//'; exit 1 ;;
esac
