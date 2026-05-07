#!/bin/bash
set -u

# ===== konfiguracja =====
BASE_DIR="/home/tk/Poker44-subnet-main"
CHECK_SCRIPT="$BASE_DIR/check_miner.sh"

STATE_DIR="/home/tk/.miner_monitor"
STATE_FILE="$STATE_DIR/state.env"
TMP_FILE="$STATE_DIR/last_check.txt"
RESTART_LOG="$STATE_DIR/restart_log.txt"

ALERT_COOLDOWN=14400   # 4h = 4*60*60

# ===== mapa scorer per miner (id → MODEL dla start_miner2.sh) =====
declare -A MINER_MODEL=(
  [13]=gen8lgbm [14]=gen8lgbm [15]=gen9fold15 [16]=gen9fold15 [17]=gen9fold15 [18]=gen9fold15
  [19]=gen9fold15 [21]=gen9fold15 [22]=gen9fold15 [23]=gen9fold15 [24]=gen9fold15
)

SMTP_HOST="pro.turbo-smtp.com"
SMTP_PORT="465"
SMTP_SECURE="true"

SMTP_USER="contact@etho.pl"
SMTP_PASS="m80V7Svs"

MAIL_FROM="contact@etho.pl"
MAIL_TO="tomek.kabarowski@gmail.com"

HOSTNAME_FQDN="$(hostname -f 2>/dev/null || hostname)"
NOW_TS="$(date +%s)"
NOW_HUMAN="$(date '+%Y-%m-%d %H:%M:%S %Z')"

restart_miner() {
  local wallet="$1"
  local id="$2"

  # Tylko sn126b jest obsługiwane przez start_miner2.sh
  if [[ "$wallet" != "sn126b" ]]; then
    echo "RESTART SKIP wallet=$wallet id=$id (only sn126b supported by start_miner2.sh)" >> "$RESTART_LOG"
    return 1
  fi

  local model="${MINER_MODEL[$id]:-}"
  local session="${wallet}_m${id}"

  {
    printf 'RESTART wallet=%s id=%s session=%s model=%s\n' "$wallet" "$id" "$session" "${model:-default}"
    printf '  stop:  '
    if screen -S "$session" -X quit >/dev/null 2>&1; then
      printf 'OK\n'
    else
      printf 'OK (session missing or already stopped)\n'
    fi

    sleep 2

    printf '  start: '
    if bash "$BASE_DIR/start_miner2.sh" "$id" 40 0 "$model" >> "$RESTART_LOG" 2>&1; then
      printf 'OK\n'
    else
      printf 'ERROR\n'
      return 1
    fi
  } >> "$RESTART_LOG" 2>&1

  return 0
}

# ===== przygotowanie =====
mkdir -p "$STATE_DIR"

if [ ! -x "$CHECK_SCRIPT" ]; then
  echo "Brak lub brak execute: $CHECK_SCRIPT"
  exit 2
fi

# domyślne wartości stanu
FAIL_ACTIVE=0
LAST_ALERT_TS=0

if [ -f "$STATE_FILE" ]; then
  # shellcheck disable=SC1090
  . "$STATE_FILE"
fi

# ===== sprawdzenie minerów =====
{
  #echo "=== sn126 ==="
  #"$CHECK_SCRIPT" sn126 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30
  #RC1=$?

  #echo
  echo "=== sn126b ==="
  "$CHECK_SCRIPT" sn126b 13,14,15,16,17,18,19,21,22,23,24
  RC2=$?

  echo
  echo "check_time=$NOW_HUMAN"
  echo "host=$HOSTNAME_FQDN"
} > "$TMP_FILE" 2>&1

# ===== auto-restart minerów z procs=0 =====
> "$RESTART_LOG"

while IFS= read -r line; do
  if [[ "$line" =~ ^FAIL\ wallet=([^[:space:]]+)\ id=([^[:space:]]+).*procs=0 ]]; then
    R_WALLET="${BASH_REMATCH[1]}"
    R_ID="${BASH_REMATCH[2]}"

    restart_miner "$R_WALLET" "$R_ID" || true
  fi
done < "$TMP_FILE"

if grep -q '^FAIL ' "$TMP_FILE"; then
  CURRENT_FAIL=1
else
  CURRENT_FAIL=0
fi

# ===== funkcja wysyłki maila =====
send_mail() {
  local subject="$1"
  local body_file="$2"

  {
    printf 'From: %s\n' "$MAIL_FROM"
    printf 'To: %s\n' "$MAIL_TO"
    printf 'Subject: %s\n' "$subject"
    printf 'Date: %s\n' "$(LC_ALL=C date -R)"
    printf 'MIME-Version: 1.0\n'
    printf 'Content-Type: text/plain; charset=UTF-8\n'
    printf '\n'
    cat "$body_file"
    printf '\n'
  } | curl --silent --show-error --fail \
      --url "smtps://${SMTP_HOST}:${SMTP_PORT}" \
      --ssl-reqd \
      --mail-from "$MAIL_FROM" \
      --mail-rcpt "$MAIL_TO" \
      --user "${SMTP_USER}:${SMTP_PASS}" \
      --upload-file -
}

# ===== logika alertowania =====
if [ "$CURRENT_FAIL" -eq 0 ]; then
  # wszystko wróciło do normy -> reset stanu
  cat > "$STATE_FILE" <<EOF
FAIL_ACTIVE=0
LAST_ALERT_TS=$LAST_ALERT_TS
EOF
  exit 0
fi

# jeśli tu jesteśmy, to jest FAIL
SEND_ALERT=0

if [ "${FAIL_ACTIVE:-0}" -eq 0 ]; then
  # nowa awaria po stanie OK -> wyślij od razu
  SEND_ALERT=1
else
  # awaria nadal trwa -> wyślij ponownie dopiero po cooldownie
  ELAPSED=$((NOW_TS - ${LAST_ALERT_TS:-0}))
  if [ "$ELAPSED" -ge "$ALERT_COOLDOWN" ]; then
    SEND_ALERT=1
  fi
fi

if [ "$SEND_ALERT" -eq 1 ]; then
  MAIL_BODY="$STATE_DIR/mail_body.txt"
  SUBJECT="[ALERT] Miner problem on ${HOSTNAME_FQDN}"

  {
    echo "Wykryto problem z minerami."
    echo
    echo "Host: $HOSTNAME_FQDN"
    echo "Czas: $NOW_HUMAN"
    echo
    echo "Szczegóły:"
    echo "----------------------------------------"
    cat "$TMP_FILE"
    echo "----------------------------------------"
    if [ -s "$RESTART_LOG" ]; then
      echo
      echo "Próby automatycznego restartu:"
      echo "----------------------------------------"
      cat "$RESTART_LOG"
      echo "----------------------------------------"
    fi
  } > "$MAIL_BODY"

  if send_mail "$SUBJECT" "$MAIL_BODY"; then
    cat > "$STATE_FILE" <<EOF
FAIL_ACTIVE=1
LAST_ALERT_TS=$NOW_TS
EOF
    exit 1
  else
    # awaria jest aktywna nawet jeśli mail się nie wysłał
    cat > "$STATE_FILE" <<EOF
FAIL_ACTIVE=1
LAST_ALERT_TS=${LAST_ALERT_TS:-0}
EOF
    exit 3
  fi
else
  # awaria trwa, ale jesteśmy w cooldownie
  cat > "$STATE_FILE" <<EOF
FAIL_ACTIVE=1
LAST_ALERT_TS=${LAST_ALERT_TS:-0}
EOF
  exit 1
fi
