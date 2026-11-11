#!/usr/bin/env bash
# An isolated Linux bridge with a separate network namespace per zone and coordinator.
set -euo pipefail

prefix="g5${BASHPID}"
bridge="${prefix}b"
subnet=198.18.52.0/24
owned_namespaces=()
forward_rule=0
placement=''
cleanup() {
  for ns in "${owned_namespaces[@]}"; do
    sudo -n ip netns pids "$ns" | xargs -r sudo -n kill -TERM -- || true
    sudo -n ip netns del "$ns" || true
  done
  if (( forward_rule )); then
    sudo -n iptables -w -D FORWARD -i "$bridge" -o "$bridge" -s "$subnet" -d "$subnet" -j ACCEPT || true
  fi
  sudo -n ip link del "$bridge" || true
  if [[ -n "$placement" ]]; then rm -f -- "$placement"; fi
}

sudo -n ip link add "$bridge" type bridge
trap cleanup EXIT
sudo -n ip addr add 198.18.52.1/24 dev "$bridge"
sudo -n ip link set "$bridge" up
sudo -n iptables -w -I FORWARD -i "$bridge" -o "$bridge" -s "$subnet" -d "$subnet" -j ACCEPT
forward_rule=1

placement=$(mktemp)
{
  printf '{\n'
  groups=(root zone-0 zone-1 zone-2 zone-3)
  for i in "${!groups[@]}"; do
    ns="${prefix}n${i}"
    veth="${prefix}v${i}"
    peer="${prefix}p${i}"
    address="198.18.52.$((i + 2))"
    sudo -n ip netns add "$ns"
    owned_namespaces+=("$ns")
    sudo -n ip link add "$veth" type veth peer name "$peer" netns "$ns"
    sudo -n ip link set "$veth" master "$bridge"
    sudo -n ip link set "$veth" up
    sudo -n ip -n "$ns" link set lo up
    sudo -n ip -n "$ns" addr add "$address/24" dev "$peer"
    sudo -n ip -n "$ns" link set "$peer" up
    if (( i > 0 )); then printf ',\n'; fi
    printf '  "%s": {"namespace": "%s", "address": "%s"}' "${groups[$i]}" "$ns" "$address"
  done
  printf '\n}\n'
} > "$placement"
python -m experiments.distributed_5g.live --namespace-placement "$placement" "$@"
