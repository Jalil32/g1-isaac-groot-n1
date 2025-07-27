"""
ZMQ Image Client - Receives and displays images from the ImageServer
"""

import cv2
import zmq
import numpy as np
import time

class ImageClient:
    def __init__(self, server_address="tcp://localhost:5555"):
        """
        Initialize ZMQ client to receive images
        """
        print(f"[Image Client] Connecting to image server at {server_address}")

        # Set up ZeroMQ context and socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(server_address)

        # Subscribe to all messages (empty string means all)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")

        # Set receive timeout to avoid blocking forever
        self.socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout

        print("[Image Client] Connected and ready to receive images")

    def receive_and_display(self):
        """
        Continuously receive and display images
        """
        print("[Image Client] Starting to receive images... Press 'q' or ESC to quit")

        frame_count = 0
        start_time = time.time()

        try:
            while True:
                try:
                    # Receive image data
                    message = self.socket.recv()

                    # Decode JPEG image
                    nparr = np.frombuffer(message, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if img is not None:
                        # Extract head camera (assuming it's the first third of the concatenated image)
                        height, width = img.shape[:2]
                        head_width = width // 3  # Assuming 3 cameras side by side
                        head_camera = img[:, 0:head_width]  # Extract leftmost third

                        # Resize to 640x480
                        head_camera_resized = cv2.resize(head_camera, (640, 480))

                        # Display the resized head camera
                        cv2.imshow('Head Camera Only (640x480)', head_camera_resized)

                        # Calculate and display FPS occasionally
                        frame_count += 1
                        if frame_count % 30 == 0:
                            elapsed = time.time() - start_time
                            fps = frame_count / elapsed
                            print(f"[Image Client] Receiving at {fps:.2f} FPS")

                        # Check for quit key
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == 27:  # 'q' or ESC
                            print("[Image Client] Quit key pressed")
                            break
                    else:
                        print("[Image Client] Failed to decode image")

                except zmq.Again:
                    # Timeout occurred, check if we should continue
                    print("[Image Client] No message received (timeout)")
                    continue

        except KeyboardInterrupt:
            print("\n[Image Client] Interrupted by user")
        finally:
            self.close()

    def close(self):
        """
        Clean up resources
        """
        cv2.destroyAllWindows()
        self.socket.close()
        self.context.term()
        print("[Image Client] Client closed")

if __name__ == "__main__":
    # Create and run the client
    client = ImageClient()
    client.receive_and_display()
