#!/usr/bin/env bash
# A shell, or one command, in the Jetson timing image with this checkout at /ws.
#   tools/jetson_bench/enter.sh                                   # a shell
#   tools/jetson_bench/enter.sh tools/jetson_bench/run_suite.sh orin_nx
# Runs as you (files keep your ownership), with HOME somewhere writable (your
# UID may have no account in the image), and the image's id in PAT_IMAGE_ID so
# every manifest records which image produced it.
set -euo pipefail
cd "$(dirname "$0")/../.."
IMG=${PAT_IMAGE:-mujoco-pat:jetson}
ID=$(docker image inspect -f '{{.Id}}' "$IMG")
TTY=()
[ -t 0 ] && TTY=(-it)
[ $# -eq 0 ] && set -- bash
exec docker run --rm "${TTY[@]}" \
    --user "$(id -u):$(id -g)" -e HOME=/tmp -e PAT_IMAGE_ID="$ID" -e JOBS -e WORKERS \
    -v "$PWD":/ws -w /ws "$IMG" "$@"
