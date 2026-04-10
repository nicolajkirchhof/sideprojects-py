#!/bin/bash
# Ensures Azure SQL is ready before starting the backend:
#   1. Resumes the database if serverless auto-pause kicked in
#   2. Whitelists the current public IP in the server firewall
#
# Multi-tenant isolation: this script uses a dedicated AZURE_CONFIG_DIR
# (~/.azure-personal by default) so the personal-tenant token cache is
# fully isolated from the work tenant. Set it up once with:
#
#     AZURE_CONFIG_DIR=~/.azure-personal \
#       az login --tenant 7c0a8551-ef69-47fa-8a29-1afdd2cbe714
#
# After that the script runs without ever touching the globally-active
# Azure CLI session (k9s, kubectl, work tenant, etc.).
#
# Usage: bash tradelog/admin/ensure-azure.sh [database_name]
#   database_name defaults to tradelog_stage
#   override PERSONAL_SUB env var if the personal subscription has a
#   different display name or id.

set -euo pipefail

export AZURE_CONFIG_DIR="${AZURE_CONFIG_DIR:-$HOME/.azure-personal}"

SERVER=ocin
RG=tradelog
DB="${1:-tradelog_stage}"
RULE_NAME="dev-machine"
PERSONAL_SUB="${PERSONAL_SUB:-Azure subscription 1}"

az_sql() {
  az "$@" --subscription "$PERSONAL_SUB"
}

# --- 1. Resume database if paused ---
STATUS=$(az_sql sql db show --server "$SERVER" -g "$RG" -n "$DB" --query status -o tsv)
if [ "$STATUS" = "Paused" ]; then
  echo "Resuming $DB..."
  az_sql sql db update --server "$SERVER" -g "$RG" -n "$DB" --auto-pause-delay 60 -o none
  while [ "$(az_sql sql db show --server "$SERVER" -g "$RG" -n "$DB" --query status -o tsv)" != "Online" ]; do
    sleep 5
  done
  echo "$DB is online"
else
  echo "$DB already online ($STATUS)"
fi

# --- 2. Whitelist current public IP ---
MY_IP=$(curl -s https://ifconfig.me)
EXISTING=$(az_sql sql server firewall-rule show --server "$SERVER" -g "$RG" -n "$RULE_NAME" --query startIpAddress -o tsv 2>/dev/null || echo "")

if [ "$EXISTING" = "$MY_IP" ]; then
  echo "IP $MY_IP already whitelisted"
else
  echo "Updating firewall rule to $MY_IP..."
  az_sql sql server firewall-rule create --server "$SERVER" -g "$RG" -n "$RULE_NAME" \
    --start-ip-address "$MY_IP" --end-ip-address "$MY_IP" -o none
  echo "IP $MY_IP whitelisted"
fi