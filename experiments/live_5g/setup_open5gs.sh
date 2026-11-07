#!/usr/bin/env bash
# Isolated GitHub runner only. Build the exact source used by the dissertation.
set -euo pipefail
MODE=${1:-build}
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  ca-certificates git build-essential meson ninja-build cmake flex bison pkg-config \
  libsctp-dev libc-ares-dev libgnutls28-dev libgcrypt20-dev libssl-dev libidn11-dev \
  libmongoc-dev libbson-dev libyaml-dev libnghttp2-dev libmicrohttpd-dev \
  libcurl4-gnutls-dev libtins-dev libtalloc-dev tcpdump
if [[ "$MODE" == build ]]; then
  git clone --branch v2.7.6 --depth 1 https://github.com/open5gs/open5gs.git open5gs-source
  test "$(git -C open5gs-source rev-parse HEAD)" = d9d3abdd480be96fac3bc8a997e83446648763ca
  meson setup open5gs-source/build open5gs-source --prefix=/opt/open5gs
  ninja -C open5gs-source/build -j 3
  sudo ninja -C open5gs-source/build install
  sudo tar -C /opt -czf open5gs-2.7.6.tar.gz open5gs
else
  sudo tar -C /opt -xzf open5gs-2.7.6.tar.gz
  sudo mkdir -p /etc/open5gs /opt/open5gs/var/log/open5gs
  sudo cp /opt/open5gs/etc/open5gs/*.yaml /etc/open5gs/
  # A profile uses distinct address/service clients. The default pool of 64
  # is exhausted by 32 actors plus the native Core; provision every NF equally.
  sudo sed -i 's/^#    peer: 64/    peer: 512/' /etc/open5gs/*.yaml
  for nf in nrf scp udr udm ausf pcf nssf smf amf; do
    sudo tee "/etc/systemd/system/open5gs-${nf}d.service" >/dev/null <<EOF
[Unit]
Description=Isolated experiment Open5GS ${nf}
After=network.target
[Service]
ExecStart=/opt/open5gs/bin/open5gs-${nf}d -c /etc/open5gs/${nf}.yaml
Restart=on-failure
RestartSec=2
[Install]
WantedBy=multi-user.target
EOF
  done
  sudo systemctl daemon-reload
  for nf in nrf scp udr udm ausf pcf nssf smf amf; do
    sudo systemctl start "open5gs-${nf}d"
  done
  sleep 15
  /opt/open5gs/bin/open5gs-nrfd -v
fi
