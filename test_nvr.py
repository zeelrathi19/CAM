import cv2

print("=" * 60)
print("CP Plus NVR RTSP Connection Test")
print("=" * 60)

# REPLACE WITH YOUR ACTUAL VALUES
IP = "192.168.1.245"        # TODO: Replace with your NVR IP
USERNAME = "admin"          # TODO: Replace with your username
PASSWORD = "Team@1703#dpa"  # TODO: Replace with your password
CHANNEL = 1                 # TODO: Replace with camera channel (1, 2, 3, etc.)

# Construct RTSP URL
url = f"rtsp://{USERNAME}:{PASSWORD}@{IP}:554/cam/realmonitor?channel={CHANNEL}&subtype=1"

print(f"\nTesting connection to:")
print(f"  IP Address: {IP}")
print(f"  Username: {USERNAME}")
print(f"  Password: {'*' * len(PASSWORD)}")
print(f"  Channel: {CHANNEL}")
print(f"\nConnecting...\n")

# Try to connect
cap = cv2.VideoCapture(url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Give it a few seconds to connect
import time
time.sleep(2)

ret, frame = cap.read()

if ret:
    print("✅ SUCCESS! Connected to camera")
    print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")
    print(f"   Channels: {frame.shape[2]}")
    print(f"\n📹 Your RTSP URL is:")
    print(f"   rtsp://{USERNAME}:{PASSWORD}@{IP}:554/cam/realmonitor?channel={CHANNEL}&subtype=1")
    print(f"\n✅ You can use this URL in the Streamlit app!")

    # Optional: Save a test frame
    cv2.imwrite("test_frame.jpg", frame)
    print(f"\n💾 Test frame saved as 'test_frame.jpg'")

else:
    print("❌ FAILED to connect!")
    print(f"\nTroubleshooting steps:")
    print(f"  1. Verify IP address is correct: {IP}")
    print(f"  2. Check username and password")
    print(f"  3. Ensure camera is connected to channel {CHANNEL}")
    print(f"  4. Check if NVR is powered on and accessible")
    print(f"  5. Try accessing NVR web interface: http://{IP}")
    print(f"\nTry different RTSP URL formats:")
    print(f"  Format 1: rtsp://{USERNAME}:{PASSWORD}@{IP}:554/cam/realmonitor?channel={CHANNEL}&subtype=0")
    print(f"  Format 2: rtsp://{USERNAME}:{PASSWORD}@{IP}:554/ch0{CHANNEL}/0")
    print(f"  Format 3: rtsp://{USERNAME}:{PASSWORD}@{IP}:8554/cam/realmonitor?channel={CHANNEL}&subtype=1")

cap.release()
print("\n" + "=" * 60)
