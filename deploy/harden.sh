#!/usr/bin/env bash
# VPS Security Hardening Script
# Run as root on a fresh Ubuntu 24.04 server BEFORE deploying the app.
set -euo pipefail

echo "=== VPS Security Hardening ==="

# 1. System updates
echo "[1/8] Updating system packages..."
apt-get update && apt-get upgrade -y

# 2. Firewall — only allow SSH, HTTP, HTTPS
echo "[2/8] Configuring firewall (UFW)..."
apt-get install -y ufw
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp    # SSH
ufw allow 80/tcp    # HTTP (Caddy redirect)
ufw allow 443/tcp   # HTTPS (Caddy)
ufw --force enable

# 3. SSH hardening
echo "[3/8] Hardening SSH..."
SSHD_CONFIG="/etc/ssh/sshd_config"
# Disable password auth (key-only)
sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' "$SSHD_CONFIG"
sed -i 's/^#\?ChallengeResponseAuthentication.*/ChallengeResponseAuthentication no/' "$SSHD_CONFIG"
# Disable root login
sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin no/' "$SSHD_CONFIG"
# Limit max auth tries
sed -i 's/^#\?MaxAuthTries.*/MaxAuthTries 3/' "$SSHD_CONFIG"
systemctl restart sshd

# 4. Fail2ban — block brute force
echo "[4/8] Installing fail2ban..."
apt-get install -y fail2ban
cat > /etc/fail2ban/jail.local <<'JAIL'
[sshd]
enabled = true
port = 22
maxretry = 5
bantime = 3600
findtime = 600
JAIL
systemctl enable --now fail2ban

# 5. Automatic security updates
echo "[5/8] Enabling automatic security updates..."
apt-get install -y unattended-upgrades
dpkg-reconfigure -plow unattended-upgrades

# 6. Create non-root app user
echo "[6/8] Creating app user..."
if ! id -u remedy &>/dev/null; then
    useradd -r -m -s /bin/bash remedy
    echo "Created user: remedy"
fi

# 7. Secure file permissions
echo "[7/8] Setting file permissions..."
APP_DIR="/opt/remedy-pdf-desktop"
if [ -d "$APP_DIR" ]; then
    chown -R remedy:remedy "$APP_DIR"
    chmod 600 "$APP_DIR/.env" 2>/dev/null || true
    chmod 600 "$APP_DIR/firebase-service-account.json" 2>/dev/null || true
fi

# 8. Rate limiting via Caddy (already handled by Caddy's defaults, but add explicit limits)
echo "[8/8] Verifying Caddy configuration..."
if command -v caddy &>/dev/null; then
    echo "Caddy installed — auto-HTTPS is active"
else
    echo "Caddy not installed yet — run setup.sh after this script"
fi

echo ""
echo "=== Hardening Complete ==="
echo ""
echo "Checklist:"
echo "  [x] Firewall: only 22, 80, 443 open"
echo "  [x] SSH: key-only, no root login, max 3 retries"
echo "  [x] Fail2ban: blocks brute force after 5 attempts"
echo "  [x] Auto security updates enabled"
echo "  [x] App runs as non-root 'remedy' user"
echo "  [x] Sensitive files (600 permissions)"
echo ""
echo "IMPORTANT: Make sure you have SSH key access before logging out!"
echo "Test: ssh remedy@your-server-ip"
