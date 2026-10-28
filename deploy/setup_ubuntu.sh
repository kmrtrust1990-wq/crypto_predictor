#!/usr/bin/env bash
# Ubuntu/Debian VPS-এ এক কমান্ডে ড্যাশবোর্ড চালু করে।
#
#   শুধু IP দিয়ে:           sudo bash deploy/setup_ubuntu.sh
#   ডোমেইন + HTTPS সহ:     sudo bash deploy/setup_ubuntu.sh predictor.example.com you@example.com
#
# আবার চালালে নতুন কোড দিয়ে কন্টেইনার নতুন করে তৈরি হয়; .env (পাসওয়ার্ড) যেমন ছিল তেমন থাকে।
set -euo pipefail

DOMAIN="${1:-}"
EMAIL="${2:-}"
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$APP_DIR"

say() { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }
fail() { printf '\n\033[1;31m✘ %s\033[0m\n' "$*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || fail "sudo দিয়ে চালান:  sudo bash deploy/setup_ubuntu.sh"
[[ -z "$DOMAIN" || -n "$EMAIL" ]] || fail "ডোমেইনের সাথে ইমেইলও দিন (HTTPS সার্টিফিকেটের জন্য)"

set_env() {  # .env-এ KEY=VALUE বসানো বা বদলানো
  if grep -q "^$1=" .env; then sed -i "s|^$1=.*|$1=$2|" .env; else echo "$1=$2" >> .env; fi
}
get_env() { grep -E "^$1=" .env | tail -1 | cut -d= -f2- || true; }

# ১. Docker
if ! command -v docker >/dev/null 2>&1; then
  say "Docker ইনস্টল হচ্ছে"
  curl -fsSL https://get.docker.com | sh
fi
systemctl enable --now docker >/dev/null 2>&1 || true
docker compose version >/dev/null 2>&1 || fail "docker compose পাওয়া যায়নি — 'apt-get install docker-compose-plugin' চালিয়ে আবার চেষ্টা করুন"

# ২. .env — প্রথমবার হলে র‍্যান্ডম পাসওয়ার্ড
[[ -f .env ]] || cp .env.example .env
if [[ -z "$(get_env APP_PASSWORD)" || "$(get_env APP_PASSWORD)" == "change-me" ]]; then
  set_env APP_PASSWORD "$(openssl rand -base64 24 | tr -dc 'A-Za-z0-9' | head -c 16)"
  say "নতুন পাসওয়ার্ড তৈরি হয়েছে (.env ফাইলে সংরক্ষিত)"
fi
PORT="$(get_env HOST_PORT)"
PORT="${PORT:-8501}"

# ৩. ডোমেইন থাকলে nginx রিভার্স প্রক্সি
if [[ -n "$DOMAIN" ]]; then
  set_env BIND_ADDR 127.0.0.1
  say "nginx ও certbot ইনস্টল হচ্ছে"
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nginx certbot python3-certbot-nginx
  sed -e "s/YOUR_DOMAIN/${DOMAIN}/g" -e "s/127\.0\.0\.1:8501/127.0.0.1:${PORT}/" \
    deploy/nginx.conf > /etc/nginx/sites-available/crypto-predictor
  ln -sf /etc/nginx/sites-available/crypto-predictor /etc/nginx/sites-enabled/crypto-predictor
  rm -f /etc/nginx/sites-enabled/default
  nginx -t
  systemctl reload nginx
  if command -v ufw >/dev/null 2>&1 && ufw status | grep -q "Status: active"; then
    ufw allow 'Nginx Full' >/dev/null
  fi
else
  set_env BIND_ADDR 0.0.0.0
fi

# ৪. ইমেজ তৈরি ও চালু
say "Docker ইমেজ তৈরি ও চালু হচ্ছে (প্রথমবার কয়েক মিনিট লাগবে)"
docker compose up -d --build

say "ড্যাশবোর্ড চালু হওয়ার অপেক্ষা"
for _ in $(seq 1 60); do
  status="$(docker inspect --format '{{.State.Health.Status}}' crypto-predictor 2>/dev/null || true)"
  [[ "$status" == "healthy" ]] && break
  sleep 2
done
[[ "$status" == "healthy" ]] || fail "ড্যাশবোর্ড চালু হয়নি — 'docker compose logs' দেখুন"

# ৫. HTTPS সার্টিফিকেট
if [[ -n "$DOMAIN" ]]; then
  say "HTTPS সার্টিফিকেট নেওয়া হচ্ছে"
  if ! certbot --nginx -d "$DOMAIN" -m "$EMAIL" --agree-tos --non-interactive --redirect; then
    printf '\n⚠ সার্টিফিকেট নেওয়া যায়নি। ডোমেইনের DNS A record এই সার্ভারের IP-তে আছে কিনা দেখুন, তারপর স্ক্রিপ্টটি আবার চালান।\n'
  fi
  URL="https://${DOMAIN}"
else
  IP="$(curl -fsS4 --max-time 5 https://ifconfig.me 2>/dev/null || hostname -I | awk '{print $1}')"
  URL="http://${IP}:${PORT}"
fi

cat <<EOF

────────────────────────────────────────────
 ✅ ড্যাশবোর্ড চালু হয়েছে
    ঠিকানা   : ${URL}
    পাসওয়ার্ড : $(get_env APP_PASSWORD)
────────────────────────────────────────────
 ক্লাউড প্রোভাইডারের ফায়ারওয়ালে (Security Group / Firewall) পোর্ট
 $([[ -n "$DOMAIN" ]] && echo "80 ও 443" || echo "$PORT") খোলা থাকতে হবে।
EOF
