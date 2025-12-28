#!/usr/bin/env python3
import subprocess
import time
import sys

print("🤖 BTC Elite Trader Expert System Status")
print("="*60)

# Check if paper trading is running
try:
    result = subprocess.run(['pgrep', '-f', 'run_paper.py'], capture_output=True, text=True)
    if result.stdout.strip():
        print("✅ Paper trading is RUNNING (PID: {})".format(result.stdout.strip()))
        print("\nTo view logs in real-time:")
        print("  tail -f logs/paper_trading.log")
        print("\nTo stop:")
        print("  kill {}".format(result.stdout.strip()))
    else:
        print("❌ Paper trading is NOT running")
        print("\nTo start:")
        print("  cd /root/bit-trand")
        print("  source venv/bin/activate")
        print("  python scripts/run_paper.py &")
except Exception as e:
    print(f"Error checking status: {e}")

print("\n" + "="*60)
print("Key differences between scripts:")
print("  btc_trader.py       = OLD backtest (-99% ROI)")
print("  scripts/run_paper.py = NEW expert system (regime filter)")
print("\nThe expert system:")
print("  - Detects market regimes (bull/bear/ranging)")
print("  - Sits out bad markets to protect capital")
print("  - Uses risk management with position sizing")
print("  - Sends alerts to Telegram")