#!/usr/bin/env bash
set -euo pipefail

WALLET_NAME="${1:-}"
IDS_ARG="${2:-}"

if [ -z "$WALLET_NAME" ]; then
  echo "Użycie: $0 WALLET_NAME [N,M,O,P]"
  exit 1
fi

case "$WALLET_NAME" in
  sn126)
    BASE_PORT=11080
    ;;
  sn126b)
    BASE_PORT=12080
    ;;
  *)
    echo "Nieznany WALLET_NAME: $WALLET_NAME"
    echo "Obsługiwane: sn126, sn126b"
    exit 1
    ;;
esac

if [ -n "$IDS_ARG" ]; then
  IFS=',' read -ra IDS <<< "$IDS_ARG"
else
  if [ "$WALLET_NAME" = "sn126b" ]; then
    IDS=($(seq 1 10))
  else
    IDS=($(seq 1 30))
  fi
fi

FAIL=0

for raw in "${IDS[@]}"; do
  I="$(echo "$raw" | xargs)"
  [ -z "$I" ] && continue
  if ! [[ "$I" =~ ^[0-9]+$ ]]; then
    echo "WARN invalid id='$I' (pomijam)"
    continue
  fi

  SESSION="${WALLET_NAME}_m${I}"
  PORT=$((BASE_PORT + I))

  SCREEN_COUNT=$(screen -list | grep -c "\.${SESSION}[[:space:]]" || true)

  PROC_COUNT=$(ps -eo pid=,comm=,args= | awk -v wallet="$WALLET_NAME" -v hk="hk${I}" -v port="$PORT" '
    ($2 == "python" || $2 == "python3") {
      if ($0 !~ /neurons\/miner\.py/ && $0 !~ /-m[[:space:]]+neurons\.miner/) next
      if ($0 !~ ("--wallet.name(=|[[:space:]])" wallet "([[:space:]]|$)")) next
      if ($0 !~ ("--wallet.hotkey(=|[[:space:]])" hk "([[:space:]]|$)")) next
      if ($0 !~ ("--axon.port(=|[[:space:]])" port "([[:space:]]|$)")) next
      if ($0 !~ /--netuid(=|[[:space:]])126([[:space:]]|$)/) next
      if ($0 !~ /--subtensor\.network(=|[[:space:]])finney([[:space:]]|$)/) next
      count++
    }
    END { print count+0 }
  ')

  if [ "$SCREEN_COUNT" -eq 1 ] && [ "$PROC_COUNT" -eq 1 ]; then
    echo "OK   wallet=${WALLET_NAME} id=${I} session=${SESSION} hotkey=hk${I} port=${PORT}"
  else
    echo "FAIL wallet=${WALLET_NAME} id=${I} session=${SESSION} hotkey=hk${I} port=${PORT} screens=${SCREEN_COUNT} procs=${PROC_COUNT}"
    FAIL=1
  fi
done

exit $FAIL