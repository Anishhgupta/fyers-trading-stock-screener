"""
Standalone Fyers connectivity test. Run AFTER fyers_login.py has given
you today's access token and you've saved it into .env.

Does nothing except confirm login works and print your profile -- no
market data, no trading, no risk.

Usage:
    python test_fyers_login.py
"""
from dotenv import load_dotenv
load_dotenv()

from brokers.base import get_env


def main():
    client_id = get_env("FYERS_CLIENT_ID")
    access_token = get_env("FYERS_ACCESS_TOKEN")
    print(f"Client ID: {client_id}")
    print(f"Access token: {access_token[:12]}{'*' * 10}")

    from fyers_apiv3 import fyersModel

    fyers = fyersModel.FyersModel(
        client_id=client_id, is_async=False, token=access_token, log_path=""
    )
    profile = fyers.get_profile()

    if profile.get("s") != "ok":
        print("\nLOGIN CHECK FAILED")
        print(profile)
        print(
            "\nMost likely cause: today's access token has expired or is "
            "wrong -- re-run fyers_login.py to generate a fresh one. "
            "Fyers tokens are valid for the current trading day only."
        )
        return

    print("\nLOGIN CHECK OK")
    data = profile.get("data", {})
    print(f"  Name: {data.get('name')}")
    print(f"  Fyers ID: {data.get('fy_id')}")
    print(f"  Email: {data.get('email_id')}")
    print("\nCredentials are working. You can now try:")
    print("  python main.py --broker fyers --duration 30")


if __name__ == "__main__":
    main()
