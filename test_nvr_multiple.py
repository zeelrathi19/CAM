import cv2
import time

print("=" * 70)
print("CP Plus NVR RTSP Connection Test - Multiple Formats")
print("=" * 70)

# REPLACE WITH YOUR ACTUAL VALUES
IP = "192.168.1.64"        # Your NVR IP
USERNAME = "admin"          # Your username
PASSWORD = "your_password"  # Your actual password (IMPORTANT: Replace this!)
CHANNEL = 1                 # Camera channel

# List of common CP Plus RTSP URL formats to try
url_formats = [
    # Format 1: Standard CP Plus format with sub stream (lower quality, better performance)
    f"rtsp://{USERNAME}:{PASSWORD}@{IP}:554/cam/realmonitor?channel={CHANNEL}&subtype=1",

    # Format 2: Standard CP Plus format with main stream (higher quality)
    f"rtsp://{USERNAME}:{PASSWORD}@{IP}:554/cam/realmonitor?channel={CHANNEL}&subtype=0",

    # Format 3: Alternative channel format
    f"rtsp://{USERNAME}:{PASSWORD}@{IP}:554/ch0{CHANNEL}/0",

    # Format 4: Alternative port (some NVRs use 8554)
    f"rtsp://{USERNAME}:{PASSWORD}@{IP}:8554/cam/realmonitor?channel={CHANNEL}&subtype=1",

    # Format 5: Older CP Plus format
    f"rtsp://{USERNAME}:{PASSWORD}@{IP}:554/1{CHANNEL}",

    # Format 6: Generic format
    f"rtsp://{USERNAME}:{PASSWORD}@{IP}:554/Streaming/Channels/{CHANNEL}01",
]

print(f"\nNVR Details:")
print(f"  IP Address: {IP}")
print(f"  Username: {USERNAME}")
print(f"  Password: {'*' * len(PASSWORD)}")
print(f"  Channel: {CHANNEL}")
print(f"\nTrying {len(url_formats)} different RTSP URL formats...\n")
print("-" * 70)

found_working_url = False

for i, url in enumerate(url_formats, 1):
    # Hide password in display
    display_url = url.replace(PASSWORD, '*' * len(PASSWORD))

    print(f"\n[{i}/{len(url_formats)}] Testing: {display_url}")
    print("    Connecting", end="", flush=True)

    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_TIMEOUT, 10000)  # 10 second timeout

    # Try to read a frame with visual feedback
    for j in range(10):
        print(".", end="", flush=True)
        time.sleep(0.5)

    ret, frame = cap.read()

    if ret:
        print(" ✅ SUCCESS!")
        print(f"\n{'=' * 70}")
        print(f"✅ WORKING URL FOUND!")
        print(f"{'=' * 70}")
        print(f"\nResolution: {frame.shape[1]}x{frame.shape[0]}")
        print(f"Channels: {frame.shape[2]}")
        print(f"\n📹 Your RTSP URL is:")
        print(f"{url}")
        print(f"\n💾 Saving test frame as 'test_frame_channel{CHANNEL}.jpg'...")

        cv2.imwrite(f"test_frame_channel{CHANNEL}.jpg", frame)
        print(f"✅ Frame saved successfully!")

        print(f"\n✅ Use this URL in your Streamlit app!")
        print(f"{'=' * 70}\n")

        found_working_url = True
        cap.release()
        break
    else:
        print(" ❌ Failed")
        cap.release()

if not found_working_url:
    print(f"\n{'=' * 70}")
    print("❌ NONE OF THE URL FORMATS WORKED")
    print(f"{'=' * 70}")
    print("\n🔍 Troubleshooting Steps:")
    print("\n1. VERIFY NETWORK CONNECTION:")
    print(f"   Run: ping {IP}")
    print(f"   The NVR must be reachable on your network")

    print("\n2. CHECK WEB INTERFACE:")
    print(f"   Open browser: http://{IP}")
    print(f"   Can you login with username '{USERNAME}' and your password?")

    print("\n3. VERIFY CREDENTIALS:")
    print(f"   - Username is correct: '{USERNAME}'")
    print(f"   - Password is correct (you entered: {'*' * len(PASSWORD)})")
    print(f"   - Try common CP Plus passwords: 'admin', 'cp123456', 'admin@123'")

    print("\n4. CHECK RTSP PORT:")
    print(f"   Run: nc -zv {IP} 554")
    print(f"   Port 554 should be open")

    print("\n5. VERIFY CHANNEL NUMBER:")
    print(f"   - Is camera actually connected to channel {CHANNEL}?")
    print(f"   - Login to NVR web interface to check active channels")

    print("\n6. CHECK NVR SETTINGS:")
    print(f"   In NVR web interface:")
    print(f"   - Setup → Network → Port → RTSP Port should be enabled (554)")
    print(f"   - Setup → Camera → Check which channels have cameras")

    print("\n7. FIREWALL/NETWORK ISSUES:")
    print(f"   - Is your Mac on the same network as the NVR?")
    print(f"   - Any firewall blocking RTSP port 554?")

    print(f"\n{'=' * 70}\n")
