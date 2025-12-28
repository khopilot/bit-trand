#!/bin/bash
# BTC Elite Trader - VPS Deployment Script
# Usage: ./deploy.sh [paper|testnet|live]

set -e

MODE=${1:-paper}
echo "🚀 Deploying BTC Elite Trader in $MODE mode..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found"
    echo "Copy .env.example to .env and fill in your credentials"
    exit 1
fi

# Check required env vars
source .env
if [ -z "$TELEGRAM_BOT_TOKEN" ] || [ -z "$TELEGRAM_CHAT_ID" ]; then
    echo "⚠️  Warning: Telegram credentials not set - will print to console"
fi

# For live mode, check Binance keys
if [ "$MODE" = "live" ]; then
    if [ -z "$BINANCE_API_KEY" ] || [ -z "$BINANCE_API_SECRET" ]; then
        echo "❌ Error: Binance API keys required for live mode"
        exit 1
    fi
    echo "⚠️  WARNING: Live trading mode - real money at risk!"
    read -p "Type 'CONFIRM' to proceed: " confirm
    if [ "$confirm" != "CONFIRM" ]; then
        echo "Aborted."
        exit 1
    fi
fi

# Build and start containers
echo "📦 Building Docker images..."
docker compose build

echo "🗄️ Starting database..."
docker compose up -d postgres
sleep 5

echo "🤖 Starting trading bot in $MODE mode..."
if [ "$MODE" = "paper" ]; then
    docker compose up -d btc-trader
elif [ "$MODE" = "testnet" ]; then
    docker compose run -d --name btc-trader-testnet btc-trader python scripts/run_paper.py --testnet
elif [ "$MODE" = "live" ]; then
    docker compose run -d --name btc-trader-live btc-trader python scripts/run_live.py
fi

echo "✅ Deployment complete!"
echo ""
echo "Commands:"
echo "  docker compose logs -f btc-trader  # View logs"
echo "  docker compose stop                # Stop all"
echo "  docker compose down                # Stop and remove"
echo ""
echo "Telegram bot: @geckocoinkh_bot"
