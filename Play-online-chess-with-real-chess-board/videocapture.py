from threading import Thread
from queue import Queue, Empty
from picamera2 import Picamera2
import time
import cv2
import numpy as np
import gc  # For garbage collection

class Video_capture_thread(Thread):
    def __init__(self, *args, **kwargs):
        super(Video_capture_thread, self).__init__(*args, **kwargs)
        self.queue = Queue(maxsize=2)  # Small queue size for better responsiveness
        self.running = True
        self.display_width = 800  # For resizing display
        self.picam = None  # Initialize to None
        self.frame_count = 0
        self.last_restart = time.time()
        self.fps = 0
        self.last_fps_time = time.time()
        self.temp_fix_count = 0  # Counter for temporary fixes
        
    def init_camera(self):
        try:
            print("Initializing Pi Camera...")
            self.picam = Picamera2()
            
            # Use a simpler still configuration instead of preview with dual streams
            # This is more reliable for our chess application
            camera_config = self.picam.create_still_configuration(
                main={"size": (1640, 1232), "format": "RGB888"},
                buffer_count=2
            )
            
            # Apply configuration
            self.picam.configure(camera_config)
            
            # Start camera
            self.picam.start()
            print(f"Pi Camera initialized with configuration: 1640x1232")
            time.sleep(2.5)  # Longer stabilization time for exposure adjustment
            return True
        except Exception as e:
            print(f"Camera initialization error: {str(e)}")
            # Try a fallback configuration if the first setup fails
            try:
                print("Trying fallback camera configuration...")
                # Create an even simpler configuration with lower resolution
                camera_config = self.picam.create_preview_configuration(
                    main={"size": (1280, 720), "format": "RGB888"}
                )
                self.picam.configure(camera_config)
                
                # Use more basic settings for the fallback
                self.picam.set_controls({
                    "Brightness": 0.4,
                    "ExposureTime": 66000,  # Even longer exposure for fallback
                    "AnalogueGain": 3.0     # Higher gain for fallback
                })
                
                self.picam.start()
                print("Camera initialized with fallback configuration")
                time.sleep(2.5)
                return True
            except Exception as fallback_error:
                print(f"Fallback initialization failed: {str(fallback_error)}")
                return False
    
    def show_camera_info(self):
        """Print detailed camera information for debugging"""
        if self.picam:
            try:
                camera_info = self.picam.camera_properties
                print("\n--- Camera Information ---")
                print(f"Model: {camera_info.get('Model', 'Unknown')}")
                print(f"Sensor modes: {len(self.picam.sensor_modes)}")
                for idx, mode in enumerate(self.picam.sensor_modes):
                    print(f"  Mode {idx}: {mode}")
                print(f"Current controls: {self.picam.camera_controls}")
                print("-------------------------\n")
            except Exception as e:
                print(f"Error getting camera info: {e}")
        
    def run(self):
        """
        Thread entry point that continuously captures frames from the camera.
        Removes the print statement that was spamming 'Camera capture FPS: ...'.
        """
        if not self.init_camera():
            print("Failed to initialize camera")
            return

        # Show camera info for debugging (optional)
        self.show_camera_info()
        
        # Reset counters and timestamps
        self.frame_count = 0
        self.last_restart = time.time()
        self.last_fps_time = time.time()
        self.fps = 0
        self.temp_fix_count = 0

        while self.running:
            try:
                # Increment frame count for possible FPS calculation
                self.frame_count += 1
                current_time = time.time()
                
                # Calculate FPS every 30 frames
                if self.frame_count % 30 == 0:
                    elapsed = current_time - self.last_fps_time
                    if elapsed > 0:
                        self.fps = 30 / elapsed
                        # Commented out the noisy print:
                        # print(f"Camera capture FPS: {self.fps:.1f}")
                        self.last_fps_time = current_time
                
                # Periodically restart camera to prevent drift/zoom issues (e.g., every 600 seconds)
                if current_time - self.last_restart > 600:
                    print("Performing routine camera restart...")
                    self.cleanup()
                    time.sleep(0.5)
                    self.init_camera()
                    self.last_restart = current_time
                    
                # Capture a single frame from the camera
                try:
                    frame = self.picam.capture_array()
                    if frame is None:
                        raise Exception("Captured frame is None")
                except Exception as e:
                    print(f"Frame capture failed: {e}")
                    time.sleep(0.5)
                    continue
                
                # Process frame if it's valid
                if frame is not None:
                    # If using RGB888, convert to BGR for OpenCV
                    # (Adjust if your camera or library returns something different)
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        import cv2
                        import numpy as np
                        
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        
                        # Optional brightness or HSV adjustments
                        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                        h, s, v = cv2.split(hsv)
                        v = cv2.add(v, 20)  # Increase brightness by constant
                        hsv = cv2.merge([h, s, v])
                        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                    
                    # Resize frame for display or processing
                    aspect_ratio = frame.shape[1] / frame.shape[0]
                    display_height = int(self.display_width / aspect_ratio)
                    import cv2
                    display_frame = cv2.resize(frame, (self.display_width, display_height),
                                               interpolation=cv2.INTER_AREA)
                    
                    # Put the new frame in the queue, dropping older ones if full
                    if self.queue.full():
                        try:
                            self.queue.get_nowait()
                        except:
                            pass
                    self.queue.put(display_frame)
                
                # Periodically do camera checks or forced garbage collection
                self.check_and_fix_camera(current_time)
                if self.frame_count % 100 == 0:
                    import gc
                    gc.collect()
                
                # Lower framerate to ~20 FPS (adjust as needed)
                time.sleep(0.05)

            except Exception as e:
                print(f"Frame capture error: {e}")
                time.sleep(0.5)
                
                # Attempt camera recovery if repeated errors occur
                self.temp_fix_count += 1
                if self.temp_fix_count > 3:
                    print("Multiple errors detected - attempting camera recovery...")
                    try:
                        self.cleanup()
                        time.sleep(1)
                        self.init_camera()
                        self.last_restart = time.time()
                        self.temp_fix_count = 0
                    except Exception as recovery_error:
                        print(f"Recovery attempt failed: {recovery_error}")

        # Clean up when thread finishes
        self.cleanup()

    
    def check_and_fix_camera(self, current_time):
        """Check for camera issues and apply fixes if needed"""
        # Check if we're getting frames (if not, this will help debug)
        if self.frame_count % 150 == 0:  # Every ~150 frames
            try:
                # Get quick CPU temperature to ensure we're not overheating
                import subprocess
                temp_output = subprocess.check_output(['vcgencmd', 'measure_temp']).decode()
                temp = float(temp_output.replace('temp=', '').replace('\'C', ''))
                
                if temp > 80:
                    print(f"WARNING: High CPU temperature: {temp}°C - risk of thermal throttling!")
                elif temp > 70:  
                    print(f"Note: Elevated CPU temperature: {temp}°C")
                
                # If queue is repeatedly empty, there might be an issue
                if self.queue.empty() and current_time - self.last_restart > 30:
                    print("Warning: Frame queue empty - possible camera stall")
                    self.temp_fix_count += 1
                    
            except Exception as e:
                # Non-critical, just log it
                print(f"Info check error: {e}")
    
    def get_frame(self):
        """Get the latest frame from the queue"""
        try:
            return self.queue.get(timeout=0.5)  # Shorter timeout for better responsiveness
        except Empty:
            return None
        except Exception as e:
            print(f"Error getting frame: {e}")
            return None
        
    def show_frame(self, window_name='Camera Feed'):
        """Display the latest frame in a window"""
        frame = self.get_frame()
        if frame is not None:
            cv2.imshow(window_name, frame)
            return cv2.waitKey(1) & 0xFF
        return 0
        
    def cleanup(self):
        """Properly clean up resources"""
        if self.picam is not None:
            try:
                self.picam.stop()
                self.picam.close()
                print("Pi Camera closed")
                self.picam = None  # Clear reference to help garbage collection
            except Exception as e:
                print(f"Error closing camera: {e}")
    
    def stop(self):
        """Stop the thread and release camera resources"""
        print("Stopping video capture thread...")
        self.running = False
        # Join will happen naturally when run() finishes

# Test function to verify the camera is working properly
def test_camera():
    print("Starting camera test...")
    video_thread = Video_capture_thread()
    video_thread.daemon = True
    video_thread.start()
    
    print("Waiting for camera to initialize...")
    time.sleep(3)
    
    try:
        print("Press 'q' to exit test")
        while True:
            key = video_thread.show_frame("Camera Test")
            if key == ord('q'):
                break
            time.sleep(0.01)
    finally:
        video_thread.stop()
        cv2.destroyAllWindows()
        print("Camera test complete")

# Run this if the script is executed directly
if __name__ == "__main__":
    test_camera()
