"""
Standalone Angel One login test.

Run this FIRST, before trying `python main.py --broker angelone`, to
confirm your .env credentials actually work in isolation. It does
nothing except log in and print your profile -- no market data, no
trading, no risk.

Usage:
    python test_angelone_login.py
"""
from dotenv import load_dotenv
load_dotenv()

from brokers.base import get_env


def main():
    print("Loading credentials from .env ...")
    api_key = get_env("ANGEL_API_KEY")
    client_id = get_env("ANGEL_CLIENT_ID")
    password = get_env("ANGEL_PASSWORD")
    totp_secret = get_env("ANGEL_TOTP_SECRET")
    print(f"  Client ID: {client_id}")
    print(f"  API Key:   {api_key[:4]}{'*' * (len(api_key) - 4)}")
    print("  Password / TOTP secret: loaded (hidden)")

    print("\nGenerating TOTP code ...")
    import pyotp
    otp = pyotp.TOTP(totp_secret).now()
    print(f"  Current TOTP: {otp}")

    print("\nLogging in to Angel One SmartAPI ...")
    from SmartApi import SmartConnect

    smart = SmartConnect(api_key=api_key)
    session = smart.generateSession(client_id, password, otp)

    if not session.get("status"):
        print("\nLOGIN FAILED")
        print(session)
        print(
            "\nCommon causes: wrong password/PIN, stale TOTP secret, "
            "API key not yet active (can take a few minutes after "
            "creation), or clock drift on this machine (TOTP is time-"
            "based -- make sure your system clock is accurate)."
        )
        return

    print("\nLOGIN SUCCESSFUL")
    jwt = session["data"]["jwtToken"]
    feed_token = smart.getfeedToken()
    print(f"  JWT token acquired: {jwt[:20]}...")
    print(f"  Feed token acquired: {feed_token[:20]}...")

    profile = smart.getProfile(jwt)
    print("\nProfile:")
    print(f"  Name: {profile.get('data', {}).get('name')}")
    print(f"  Client code: {profile.get('data', {}).get('clientcode')}")
    print(f"  Email: {profile.get('data', {}).get('email')}")
    print(
        "\nCredentials are working. You can now try:\n"
        "  python main.py --broker angelone --duration 30\n"
        "(you'll still need the symbol->token map wired in before real "
        "market-depth data flows -- see angelone_client.py's docstring)"
    )


if __name__ == "__main__":
    main()
