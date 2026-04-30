#!/usr/bin/env bash
# Remedy PDF Desktop — VPS Setup Script
# Tested on Ubuntu 24.04 LTS
set -euo pipefail

APP_DIR="/opt/remedy-pdf-desktop"
APP_USER="remedy"
APP_DOMAIN="${APP_DOMAIN:-remedy-pdf-desktop.example.com}"
NODE_MAJOR_REQUIRED=22

echo "=== Remedy PDF Desktop — Server Setup ==="

# 1. System packages
echo "[1/7] Installing system packages..."
apt-get update
apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg openssl \
    python3 python3-venv python3-pip \
    ghostscript \
    default-jre-headless \
    caddy \
    git

if ! command -v node >/dev/null 2>&1 || \
    [ "$(node -p 'Number(process.versions.node.split(".")[0])')" -lt "$NODE_MAJOR_REQUIRED" ]; then
    curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR_REQUIRED}.x" | bash -
    apt-get install -y nodejs
fi

# 2. Create app user
echo "[2/7] Creating app user..."
id -u "$APP_USER" &>/dev/null || useradd -r -m -s /bin/bash "$APP_USER"

# 3. Clone or update repo
echo "[3/7] Setting up application..."
if [ -d "$APP_DIR" ]; then
    cd "$APP_DIR" && git pull
else
    git clone https://github.com/projectremedyai/remedy-pdf-desktop.git "$APP_DIR"
fi
chown -R "$APP_USER:$APP_USER" "$APP_DIR"

# 4. Python environment
echo "[4/7] Setting up Python environment..."
cd "$APP_DIR"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e "."

# 5. Environment config
echo "[5/7] Configuring environment..."
APP_KEY="${APP_API_KEY:-}"
if [ -f "$APP_DIR/.env" ] && grep -q '^APP_API_KEY=' "$APP_DIR/.env"; then
    APP_KEY="$(grep '^APP_API_KEY=' "$APP_DIR/.env" | tail -n 1 | cut -d= -f2-)"
fi
if [ -z "$APP_KEY" ]; then
    APP_KEY="$(openssl rand -hex 32)"
fi

if [ ! -f "$APP_DIR/.env" ]; then
    cp "$APP_DIR/.env.example" "$APP_DIR/.env"
fi
if grep -q '^APP_ENV=' "$APP_DIR/.env"; then
    sed -i 's/^APP_ENV=.*/APP_ENV=production/' "$APP_DIR/.env"
else
    printf '\nAPP_ENV=production\n' >> "$APP_DIR/.env"
fi
if grep -q '^APP_API_KEY=' "$APP_DIR/.env"; then
    sed -i "s/^APP_API_KEY=.*/APP_API_KEY=${APP_KEY}/" "$APP_DIR/.env"
else
    printf 'APP_API_KEY=%s\n' "$APP_KEY" >> "$APP_DIR/.env"
fi
if grep -q '^CORS_ORIGINS=' "$APP_DIR/.env"; then
    sed -i "s|^CORS_ORIGINS=.*|CORS_ORIGINS=https://${APP_DOMAIN}|" "$APP_DIR/.env"
else
    printf 'CORS_ORIGINS=https://%s\n' "$APP_DOMAIN" >> "$APP_DIR/.env"
fi
chmod 600 "$APP_DIR/.env"

mkdir -p "$APP_DIR/data/uploads" "$APP_DIR/data/output"
chown -R "$APP_USER:$APP_USER" "$APP_DIR/data"

# 6. Frontend build
echo "[6/7] Building frontend..."
cd "$APP_DIR/web"
export VITE_APP_API_KEY="$APP_KEY"
npm ci
npm run build

# 7. Install services
echo "[7/7] Installing services..."
cp "$APP_DIR/deploy/remedy-pdf-desktop.service" /etc/systemd/system/
sed "s|remedy-pdf-desktop.example.com|${APP_DOMAIN}|g" "$APP_DIR/deploy/Caddyfile" > /etc/caddy/Caddyfile
chown -R "$APP_USER:$APP_USER" "$APP_DIR"
systemctl daemon-reload
systemctl enable --now remedy-pdf-desktop
systemctl restart caddy

echo ""
echo "=== Setup complete ==="
echo "App: https://${APP_DOMAIN}"
echo "API key stored in: $APP_DIR/.env"
echo "Logs: journalctl -u remedy-pdf-desktop -f"
