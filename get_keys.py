"""
Run this ONCE to generate your Polymarket CLOB API key from your wallet private key.
Usage: python get_keys.py
Then copy the output into your .env file.
"""

import os
from dotenv import load_dotenv
load_dotenv()

WALLET_KEY = os.getenv("WALLET_PRIVATE_KEY", "")

if not WALLET_KEY or WALLET_KEY.startswith("0x_PASTE"):
    print("ERROR: Add your WALLET_PRIVATE_KEY to .env first, then re-run this script.")
    exit(1)

try:
    from py_clob_client.client import ClobClient

    host      = "https://clob.polymarket.com"
    chain_id  = 137   # Polygon

    print("Connecting to Polymarket CLOB...")
    client = ClobClient(host, key=WALLET_KEY, chain_id=chain_id)
    creds  = client.create_or_derive_api_creds()

    print("\n" + "═"*50)
    print("  Your Polymarket CLOB Credentials")
    print("═"*50)
    print(f"  CLOB_API_KEY    = {creds.api_key}")
    print(f"  CLOB_SECRET     = {creds.api_secret}")
    print(f"  CLOB_PASSPHRASE = {creds.api_passphrase}")
    print("═"*50)
    print("\nCopy these 3 values into your .env file.")
    print("Keep them secret — treat like a password.\n")

except ImportError:
    print("ERROR: Run this first:  pip install py-clob-client")
except Exception as e:
    print(f"ERROR: {e}")
    print("Check your wallet private key in .env — must start with 0x")
