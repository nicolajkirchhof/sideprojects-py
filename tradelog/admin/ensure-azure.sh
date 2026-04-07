#!/bin/bash
# Ensures Azure SQL is ready before starting the backend:
#   1. Resumes the database if serverless auto-pause kicked in
#   2. Whitelists the current public IP in the server firewall
#
# Usage: bash tradelog/admin/ensure-azure.sh [database_name]
#   database_name defaults to tradelog_stage

set -euo pipefail

SERVER=ocin
RG=tradelog
DB="${1:-tradelog_stage}"
RULE_NAME="dev-machine"

# --- 1. Resume database if paused ---
STATUS=$(az sql db show --server "$SERVER" -g "$RG" -n "$DB" --query status -o tsv)
if [ "$STATUS" = "Paused" ]; then
  echo "Resuming $DB..."
  az sql db update --server "$SERVER" -g "$RG" -n "$DB" --auto-pause-delay 60 -o none
  while [ "$(az sql db show --server "$SERVER" -g "$RG" -n "$DB" --query status -o tsv)" != "Online" ]; do
    sleep 5
  done
  echo "$DB is online"
else
  echo "$DB already online ($STATUS)"
fi

# --- 2. Whitelist current public IP ---
MY_IP=$(curl -s https://ifconfig.me)
EXISTING=$(az sql server firewall-rule show --server "$SERVER" -g "$RG" -n "$RULE_NAME" --query startIpAddress -o tsv 2>/dev/null || echo "")

if [ "$EXISTING" = "$MY_IP" ]; then
  echo "IP $MY_IP already whitelisted"
else
  echo "Updating firewall rule to $MY_IP..."
  az sql server firewall-rule create --server "$SERVER" -g "$RG" -n "$RULE_NAME" \
    --start-ip-address "$MY_IP" --end-ip-address "$MY_IP" -o none
  echo "IP $MY_IP whitelisted"
fi