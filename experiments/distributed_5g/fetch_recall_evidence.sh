#!/usr/bin/env bash
# Extract only the archived telemetry and decisions used by the recall comparison.
set -euo pipefail
receipts=false
if [[ ${1:-} == --receipts ]]; then receipts=true; shift; fi
dest=$1
shift
temp=$(mktemp -d)
trap 'rm -f "$temp/outer.zip" "$temp/capture.zip"; rmdir "$temp"' EXIT
for seed in "$@"; do
    case "$seed" in
        90401) id=14049128601; hash=cdbf09797a8db64963fe62627710aa618440a14beae527cb647cc7d0913674ef ;;
        90402) id=14049887852; hash=454c064539ce036dc386013b03eaa6cdc1b0cf9685d31c15634d4250f5742b4c ;;
        90405) id=14050881142; hash=00f2483c3001f64754ac8050af8f9fc52927d5cfde4f618e3fe71f872df25214 ;;
        90406) id=14048993238; hash=5dff3dee20bb74e9e8130d0972aef28eeb4f55b64c8d9ce7d7514202836b495d ;;
        90407) id=14049407530; hash=251b1dc6efef077eeaccfc3af374ec9dc0b27defa8713dd14e14cdb1b6ddbf9c ;;
        90408) id=14048857700; hash=90904a7a9b5c590771e1cf27958c9bb029ae91c891c3b4e80b01da3f3f81c548 ;;
        90409) id=14049716184; hash=c1636eb317549b6a52cf84f43e5c1973ac56da6f33c5c87c752c53e8cf584981 ;;
        90410) id=14049337789; hash=1fe13eeefc57f4ce4d6d8951f9322331674800075ab3c3c393c83c7fd9620cd7 ;;
        91301) id=14698299548; hash=78c2110487a6a2266d45053519780fd3df31dbaa875b333bd95478b7f625941a ;;
        91302) id=14697769587; hash=418a681ba88a78ce7cbbe641756b56777de96ff563b2de958ac7b09cc472eecf ;;
        91303) id=14698525810; hash=1d8acb14f08fc0a3a04a43794911052e588d76f01a03e0b60d0aadd73a0c6c55 ;;
        *) echo "Unknown capture $seed" >&2; exit 2 ;;
    esac
    inner=full-evidence.zip
    files=(test/manifest.json test/features.csv test/labels.csv test/distributed_predictions.jsonl)
    if [[ $seed == 913* ]]; then inner=v9-ood-full-evidence.zip; fi
    if [[ $receipts == true ]]; then
        files+=(test/control_events.jsonl)
    elif [[ $seed == 913* || $seed == 90408 ]]; then
        files+=(test/online_predictions.csv)
    fi
    gh api "repos/$GITHUB_REPOSITORY/actions/artifacts/$id/zip" > "$temp/outer.zip"
    echo "$hash  $temp/outer.zip" | sha256sum --check --status
    unzip -p "$temp/outer.zip" "$inner" > "$temp/capture.zip"
    mkdir -p "$dest/$seed"
    unzip -oq "$temp/capture.zip" "${files[@]}" -d "$dest/$seed"
    jq -e --argjson seed "$seed" '.seed == $seed and .actors == 32 and .windows == 3600' "$dest/$seed/test/manifest.json" >/dev/null
done
