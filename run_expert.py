#!/usr/bin/env python3
"""
Wrapper to run the expert system with .env loaded
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print("✅ Loaded .env file")
    
    # Verify Telegram config
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if bot_token and chat_id:
        print(f"✅ Telegram configured: Bot={bot_token[:10]}..., Chat={chat_id}")
    else:
        print("⚠️  Telegram not configured")
else:
    print("⚠️  No .env file found")

# Run the paper trading script
if len(sys.argv) > 1 and sys.argv[1] == 'validation':
    from scripts.run_validation import main
    print("\n🔍 Running Expert System Validation...")
else:
    from scripts.run_paper import main
    print("\n🤖 Running Expert Paper Trading...")

# Execute main
import asyncio
asyncio.run(main())