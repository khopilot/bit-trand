#!/bin/bash
# Start Expert Trading Bot with proper environment

cd /root/bit-trand

# Load environment variables from .env
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "✅ Loaded environment variables"
else
    echo "❌ No .env file found!"
    exit 1
fi

# Verify Telegram config
if [ -n "$TELEGRAM_BOT_TOKEN" ] && [ -n "$TELEGRAM_CHAT_ID" ]; then
    echo "✅ Telegram configured"
else
    echo "⚠️  Telegram not configured - will print to console"
fi

# Activate virtual environment
source venv/bin/activate

# Start paper trading
echo "🚀 Starting BTC Elite Trader Expert System..."
python scripts/run_paper.py