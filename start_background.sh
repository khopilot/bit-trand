#!/bin/bash
# Start Expert Trading Bot in background

cd /root/bit-trand

# Check if already running
if pgrep -f "run_paper.py" > /dev/null; then
    echo "❌ Bot is already running!"
    echo "To stop: pkill -f run_paper.py"
    exit 1
fi

# Start in background
nohup ./start_expert.sh > expert_bot.log 2>&1 &
PID=$!

echo "✅ Expert bot started with PID: $PID"
echo ""
echo "Commands:"
echo "  tail -f expert_bot.log     # View logs"
echo "  ps aux | grep run_paper    # Check status"
echo "  kill $PID                  # Stop bot"
echo ""
echo "Check Telegram for bot messages!"