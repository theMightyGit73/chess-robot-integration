import tkinter as tk
from tkinter.simpledialog import askstring
from tkinter import messagebox, ttk
import subprocess
import sys
import os
import shutil
import signal
import threading
import pickle
import platform
import time
import traceback
from datetime import datetime
import psutil  # For monitoring system resources

# ===============================
# Global variables and constants
# ===============================
running_process = None
token = ""
logs = []  # To store log history
MAX_LOG_LINES = 1000  # Maximum number of log lines to keep
app_version = "1.1.0"  # App version for tracking
config_directory = os.path.join(os.path.expanduser("~"), ".chess_robot")
LOG_DIRECTORY = os.path.join(config_directory, "logs")

# Create config directory if it doesn't exist
os.makedirs(config_directory, exist_ok=True)
os.makedirs(LOG_DIRECTORY, exist_ok=True)

# Pi-specific resolution options
PI_RESOLUTION_OPTIONS = ["Default", "640 x 480", "800 x 600", "1280 x 720", "1640 x 1232"]
PI_FPS_OPTIONS = ["Default", "5", "10", "15", "20", "30"]

# Normal resolution options for other platforms
STD_RESOLUTION_OPTIONS = ["Default", "640 x 480", "1280 x 720", "1920 x 1080", "2560 x 1440", "3840 x 2160"]
STD_FPS_OPTIONS = ["Default", "15", "24", "30", "60", "120", "144", "240"]

# File paths
LOG_FILE = os.path.join(LOG_DIRECTORY, f"chess_robot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
settings_file = os.path.join(config_directory, 'gui_settings.bin')
promotion_file = os.path.join(config_directory, 'promotion.bin')

# ===============================
# Utility Functions
# ===============================

def log_message(message, level="INFO", to_console=True):
    """Log a message to the GUI, log file, and optionally to console"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] [{level}] {message}"
    
    # Add to GUI logs
    logs.append(full_message)
    
    # Trim logs if too many
    if len(logs) > MAX_LOG_LINES:
        logs.pop(0)
    
    # Add to GUI text widget if it exists
    if 'logs_text' in globals() and logs_text:
        try:
            logs_text.insert(tk.END, full_message + "\n")
            logs_text.see(tk.END)  # Auto-scroll to end
        except Exception as e:
            print(f"Error updating GUI log: {e}")

    # Write to log file
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(full_message + "\n")
    except Exception as e:
        print(f"Error writing to log file: {e}")
    
    # Print to console if requested
    if to_console:
        print(full_message)

def is_raspberry_pi():
    """Check if running on Raspberry Pi"""
    try:
        return os.path.exists('/proc/device-tree/model') and 'Raspberry Pi' in open('/proc/device-tree/model').read()
    except Exception as e:
        log_message(f"Error checking for Raspberry Pi: {e}", level="WARNING")
        return False

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = ["opencv-python", "numpy", "pyttsx3", "python-chess"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        log_message(f"Missing dependencies: {', '.join(missing_packages)}", level="WARNING")
        return False, missing_packages
    return True, []

def get_system_info():
    """Get system information for diagnostics"""
    info = {}
    try:
        # Platform info
        info['platform'] = platform.platform()
        info['python_version'] = platform.python_version()
        
        # CPU info
        if is_raspberry_pi():
            # Get Raspberry Pi temperature
            try:
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    temp = float(f.read()) / 1000.0
                info['cpu_temp'] = f"{temp:.1f}°C"
            except Exception:
                info['cpu_temp'] = "Unknown"
        
        # Memory info
        mem = psutil.virtual_memory()
        info['memory_total'] = f"{mem.total / (1024**3):.1f} GB"
        info['memory_used'] = f"{mem.used / (1024**3):.1f} GB ({mem.percent}%)"
        
        # Disk info
        disk = psutil.disk_usage('/')
        info['disk_total'] = f"{disk.total / (1024**3):.1f} GB"
        info['disk_used'] = f"{disk.used / (1024**3):.1f} GB ({disk.percent}%)"
        
    except Exception as e:
        log_message(f"Error getting system info: {e}", level="WARNING")
    
    return info

def ensure_file_exists(filename, default_content=None):
    """Ensure a file exists, creating it with default content if necessary"""
    try:
        if not os.path.exists(filename):
            if default_content is not None:
                with open(filename, 'wb') as f:
                    pickle.dump(default_content, f)
            else:
                # Just create empty file
                with open(filename, 'w') as f:
                    pass
            return True
        return True
    except Exception as e:
        log_message(f"Error ensuring file exists ({filename}): {e}", level="ERROR")
        return False

def backup_file(filename):
    """Create a backup of the specified file"""
    if os.path.exists(filename):
        try:
            backup_name = f"{filename}.bak"
            shutil.copy2(filename, backup_name)
            log_message(f"Created backup: {backup_name}")
            return True
        except Exception as e:
            log_message(f"Error creating backup of {filename}: {e}", level="ERROR")
    return False

# ===============================
# GUI Functions
# ===============================

def show_about_dialog():
    """Show information about the app"""
    info = get_system_info()
    
    # Build the message
    about_text = f"Chess Robot v{app_version}\n\n"
    about_text += "A computer vision system for playing chess with a physical board\n\n"
    
    # Add system info
    about_text += "System Information:\n"
    for key, value in info.items():
        about_text += f"  {key.replace('_', ' ').title()}: {value}\n"
    
    # Show the dialog
    messagebox.showinfo("About Chess Robot", about_text)

def show_help_dialog():
    """Show help information"""
    help_text = """Chess Robot Help

Getting Started:
1. Connect your camera and ensure it's properly recognized.
2. Select the appropriate camera from the dropdown menu.
3. Run 'Board Calibration' with an empty chessboard.
4. Run 'Diagnostic' to check that pieces are recognized.
5. Click 'Start Game' to begin playing.

Troubleshooting:
• If camera isn't detected, try selecting "Default" and restarting.
• Ensure good lighting on your chess board.
• For Raspberry Pi, lower resolution may improve performance.
• If moves aren't detected, run "Diagnostic" to check piece recognition.

For more information, check the GitHub repository:
https://github.com/karayaman/Play-online-chess-with-real-chess-board
"""
    messagebox.showinfo("Chess Robot Help", help_text)

def clear_logs():
    """Clear the log display"""
    logs_text.delete(1.0, tk.END)
    logs.clear()
    log_message("Logs cleared", level="INFO")

def export_logs():
    """Export logs to a file"""
    try:
        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")],
            title="Export Logs"
        )
        if filename:
            with open(filename, 'w') as f:
                for log in logs:
                    f.write(log + "\n")
            log_message(f"Logs exported to {filename}", level="INFO")
    except Exception as e:
        log_message(f"Error exporting logs: {e}", level="ERROR")
        messagebox.showerror("Error", f"Failed to export logs: {e}")

def lichess():
    """Handle Lichess API token input"""
    global token
    new_token = askstring(
        "Lichess API Access Token", 
        "Please enter your Lichess API Access Token below.\n\n"
        "You can get an API token from https://lichess.org/account/oauth/token",
        initialvalue=token
    )
    if new_token is None:
        pass
    else:
        token = new_token
        log_message("Lichess API token updated", level="INFO")

def on_closing():
    """Handle window closing event"""
    global running_process
    try:
        if running_process:
            if running_process.poll() is None:
                log_message("Terminating running process", level="INFO")
                # Send SIGTERM on Unix-like systems
                if platform.system() != "Windows":
                    running_process.send_signal(signal.SIGTERM)
                # Windows fallback
                running_process.terminate()
                # Give process time to terminate
                for _ in range(10):
                    if running_process.poll() is not None:
                        break
                    time.sleep(0.1)
                # Force kill if still running
                if running_process.poll() is None:
                    if platform.system() != "Windows":
                        running_process.send_signal(signal.SIGKILL)
                    else:
                        running_process.kill()
    except Exception as e:
        log_message(f"Error terminating process: {e}", level="ERROR")
    
    try:
        save_settings()
    except Exception as e:
        log_message(f"Error saving settings: {e}", level="ERROR")
    
    # Close the window
    window.destroy()

def create_status_bar():
    """Create a status bar at the bottom of the window"""
    status_bar = ttk.Frame(window)
    status_bar.grid(row=14, column=0, columnspan=2, sticky="ew")
    
    # Status label (left-aligned)
    status_label = ttk.Label(status_bar, text="Ready", anchor=tk.W)
    status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=2)
    
    # System info (right-aligned)
    system_info = ttk.Label(status_bar, text="", anchor=tk.E)
    system_info.pack(side=tk.RIGHT, padx=5, pady=2)
    
    # Update system info periodically
    def update_system_info():
        if is_raspberry_pi():
            try:
                # Get CPU temperature
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    temp = float(f.read()) / 1000.0
                # Get memory usage
                mem = psutil.virtual_memory()
                # Update the label
                system_info.config(text=f"CPU: {temp:.1f}°C | RAM: {mem.percent}%")
            except Exception as e:
                system_info.config(text="System info unavailable")
        else:
            # For non-Pi systems, just show memory usage
            try:
                mem = psutil.virtual_memory()
                system_info.config(text=f"RAM: {mem.percent}%")
            except Exception:
                system_info.config(text="System info unavailable")
        
        # Schedule the next update
        window.after(5000, update_system_info)
    
    # Start the periodic updates
    update_system_info()
    
    return status_label, system_info

def update_status(message):
    """Update the status bar message"""
    if 'status_label' in globals() and status_label:
        status_label.config(text=message)

def create_buttons():
    """Create the main control buttons with improved styling"""
    global button_frame, start, board, diagnostic_button
    
    button_frame = ttk.Frame(window)
    button_frame.grid(row=12, column=0, columnspan=2, sticky="w", padx=5, pady=5)
    
    # Create buttons with ttk styling
    start = ttk.Button(button_frame, text="Start Game", command=start_game)
    start.grid(row=0, column=0, padx=2)
    
    board = ttk.Button(button_frame, text="Board Calibration", command=board_calibration)
    board.grid(row=0, column=1, padx=2)
    
    diagnostic_button = ttk.Button(button_frame, text="Diagnostic", command=diagnostic)
    diagnostic_button.grid(row=0, column=2, padx=2)
    
    # Add advanced options button
    advanced_button = ttk.Button(
        button_frame, 
        text="Advanced...", 
        command=lambda: show_advanced_options()
    )
    advanced_button.grid(row=0, column=3, padx=2)

def show_advanced_options():
    """Show advanced options dialog"""
    advanced_window = tk.Toplevel(window)
    advanced_window.title("Advanced Options")
    advanced_window.transient(window)  # Set as transient to main window
    advanced_window.grab_set()  # Modal dialog
    
    # Add some padding
    frame = ttk.Frame(advanced_window, padding=10)
    frame.pack(fill=tk.BOTH, expand=True)
    
    # Debug mode option
    debug_var = tk.IntVar(value=0)
    debug_check = ttk.Checkbutton(
        frame, 
        text="Enable Debug Mode", 
        variable=debug_var
    )
    debug_check.grid(row=0, column=0, sticky="w", pady=5)
    
    # Backup calibration data
    backup_button = ttk.Button(
        frame,
        text="Backup Calibration Data",
        command=lambda: backup_calibration_data()
    )
    backup_button.grid(row=1, column=0, sticky="w", pady=5)
    
    # Reset all settings
    reset_button = ttk.Button(
        frame,
        text="Reset All Settings",
        command=lambda: reset_settings(advanced_window)
    )
    reset_button.grid(row=2, column=0, sticky="w", pady=5)
    
    # Check system resources
    system_button = ttk.Button(
        frame,
        text="Check System Resources",
        command=lambda: check_resources(advanced_window)
    )
    system_button.grid(row=3, column=0, sticky="w", pady=5)
    
    # Buttons at bottom
    button_frame = ttk.Frame(frame)
    button_frame.grid(row=4, column=0, pady=10)
    
    ok_button = ttk.Button(
        button_frame, 
        text="OK", 
        command=lambda: process_advanced_options(advanced_window, debug_var.get())
    )
    ok_button.pack(side=tk.RIGHT, padx=5)
    
    cancel_button = ttk.Button(
        button_frame,
        text="Cancel",
        command=advanced_window.destroy
    )
    cancel_button.pack(side=tk.RIGHT, padx=5)
    
    # Center the window
    advanced_window.update_idletasks()
    width = advanced_window.winfo_width()
    height = advanced_window.winfo_height()
    x = (advanced_window.winfo_screenwidth() // 2) - (width // 2)
    y = (advanced_window.winfo_screenheight() // 2) - (height // 2)
    advanced_window.geometry(f'{width}x{height}+{x}+{y}')

def process_advanced_options(window, debug_enabled):
    """Process the advanced options"""
    global debug_mode
    debug_mode = debug_enabled
    log_message(f"Debug mode {'enabled' if debug_enabled else 'disabled'}")
    window.destroy()

def backup_calibration_data():
    """Backup calibration data files"""
    success = False
    try:
        # List of files to backup
        backup_files = ['constants.bin', 'promotion.bin']
        
        # Create backup directory
        backup_dir = os.path.join(config_directory, 'backups', 
                                datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(backup_dir, exist_ok=True)
        
        # Copy files
        for file in backup_files:
            if os.path.exists(file):
                shutil.copy2(file, os.path.join(backup_dir, file))
                success = True
        
        if success:
            log_message(f"Calibration data backed up to {backup_dir}")
            messagebox.showinfo("Backup Complete", 
                              f"Calibration data backed up to:\n{backup_dir}")
        else:
            log_message("No calibration files found to backup", level="WARNING")
            messagebox.showinfo("Nothing to Backup", "No calibration files found to backup.")
    except Exception as e:
        log_message(f"Error backing up calibration data: {e}", level="ERROR")
        messagebox.showerror("Backup Failed", f"Failed to backup calibration data:\n{e}")

def reset_settings(parent_window=None):
    """Reset all settings after confirmation"""
    global token
    
    if parent_window:
        result = messagebox.askyesno(
            "Confirm Reset",
            "This will reset all settings to default values.\nAre you sure?",
            parent=parent_window
        )
    else:
        result = messagebox.askyesno(
            "Confirm Reset",
            "This will reset all settings to default values.\nAre you sure?"
        )
    
    if result:
        try:
            # Reset variables
            for field in fields:
                if isinstance(field, tk.StringVar):
                    if field == camera:
                        field.set(OPTIONS[0])
                    elif field == resolution:
                        field.set(resolution_options[0])
                    elif field == fps:
                        field.set(fps_options[0])
                    elif field == voice:
                        field.set(VOICE_OPTIONS[0])
                    elif field == calibration_mode:
                        field.set(CALIBRATION_OPTIONS[0])
                    else:
                        field.set("")
                elif isinstance(field, tk.IntVar):
                    field.set(0)
                else:
                    field.set(0)
            
            # Reset token
            token = ""
            
            # Set default delay
            default_value.set(values[-1])
            
            # Reset promotion
            promotion.set(PROMOTION_OPTIONS[0])
            promotion_menu.configure(state="disabled")
            
            # Save the reset settings
            save_settings()
            
            log_message("All settings have been reset to defaults", level="INFO")
            if parent_window:
                messagebox.showinfo("Reset Complete", 
                                  "All settings have been reset to defaults.",
                                  parent=parent_window)
            else:
                messagebox.showinfo("Reset Complete", 
                                  "All settings have been reset to defaults.")
                
        except Exception as e:
            log_message(f"Error resetting settings: {e}", level="ERROR")
            if parent_window:
                messagebox.showerror("Reset Failed", 
                                   f"Failed to reset settings:\n{e}",
                                   parent=parent_window)
            else:
                messagebox.showerror("Reset Failed", 
                                   f"Failed to reset settings:\n{e}")

def check_resources(parent_window=None):
    """Check and display system resources"""
    try:
        # Get CPU info
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Get memory info
        memory = psutil.virtual_memory()
        
        # Get disk info
        disk = psutil.disk_usage('/')
        
        # Get temperature on Raspberry Pi
        temp = "N/A"
        if is_raspberry_pi():
            try:
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    temp = f"{float(f.read()) / 1000.0:.1f}°C"
            except:
                pass
        
        # Create results message
        message = "System Resources:\n\n"
        message += f"CPU Usage: {cpu_percent}%\n"
        message += f"CPU Temperature: {temp}\n\n"
        message += f"Memory Total: {memory.total / (1024**3):.1f} GB\n"
        message += f"Memory Used: {memory.used / (1024**3):.1f} GB ({memory.percent}%)\n\n"
        message += f"Disk Total: {disk.total / (1024**3):.1f} GB\n"
        message += f"Disk Used: {disk.used / (1024**3):.1f} GB ({disk.percent}%)\n"
        
        # Show warning if resources are low
        warning = ""
        if memory.percent > 90:
            warning += "WARNING: Memory usage is very high!\n"
        if disk.percent > 90:
            warning += "WARNING: Disk space is very low!\n"
        if is_raspberry_pi() and temp != "N/A" and float(temp.replace("°C", "")) > 80:
            warning += "WARNING: CPU temperature is very high!\n"
            
        if warning:
            message += f"\n{warning}"
        
        # Log the check
        log_message("System resources check completed", level="INFO")
        
        # Show in dialog
        if parent_window:
            messagebox.showinfo("System Resources", message, parent=parent_window)
        else:
            messagebox.showinfo("System Resources", message)
            
    except Exception as e:
        log_message(f"Error checking system resources: {e}", level="ERROR")
        if parent_window:
            messagebox.showerror("Resource Check Failed", 
                               f"Failed to check system resources:\n{e}",
                               parent=parent_window)
        else:
            messagebox.showerror("Resource Check Failed", 
                               f"Failed to check system resources:\n{e}")

def log_process(process, finish_message):
    """Log process output to the GUI text widget"""
    global button_frame, running_process
    
    # Replace buttons with stop button
    button_stop = ttk.Button(button_frame, text="Stop", command=stop_process)
    button_stop.grid(row=0, column=0, columnspan=4, sticky="ew")
    
    # Update status
    update_status("Process running...")
    
    # Show process output
    while True:
        try:
            output = process.stdout.readline()
            if output:
                output_str = output if isinstance(output, str) else output.decode()
                
                # Add to logs
                log_message(output_str.strip(), to_console=False)
                
                # Update GUI more frequently
                window.update_idletasks()
            
            # Check if process has ended
            if process.poll() is not None:
                # Get any remaining output
                for output in process.stdout:
                    output_str = output if isinstance(output, str) else output.decode()
                    log_message(output_str.strip(), to_console=False)
                
                log_message(finish_message)
                update_status("Ready")
                break
                
            # Short sleep to reduce CPU usage
            time.sleep(0.01)
            
        except Exception as e:
            log_message(f"Error reading process output: {e}", level="ERROR")
            break
    
    # Restore original buttons when process completes
    create_buttons()
    
    if promotion_menu.cget("state") == "normal":
        promotion.set(PROMOTION_OPTIONS[0])
        promotion_menu.configure(state="disabled")
    
    # Clear the running process reference
    running_process = None

def stop_process(ignore=None):
    """Stop the running process"""
    global running_process
    
    if running_process:
        try:
            if running_process.poll() is None:
                # Send SIGTERM on Unix-like systems
                if platform.system() != "Windows":
                    running_process.send_signal(signal.SIGTERM)
                # Windows fallback
                running_process.terminate()
                
                log_message("Process stopped by user", level="INFO")
                update_status("Process terminated")
                
                # Give process time to terminate
                for _ in range(10):
                    if running_process.poll() is not None:
                        break
                    time.sleep(0.1)
                
                # Force kill if still running
                if running_process.poll() is None:
                    log_message("Process not responding - forcing termination", level="WARNING")
                    if platform.system() != "Windows":
                        running_process.send_signal(signal.SIGKILL)
                    else:
                        running_process.kill()
        except Exception as e:
            log_message(f"Error stopping process: {e}", level="ERROR")

def start_process(executable, args, finish_message, run_with_debug=False):
    """Start a subprocess with proper error handling"""
    global running_process
    
    # Ensure no other process is running
    if running_process:
        log_message("Another process is already running", level="WARNING")
        messagebox.showwarning("Process Running", 
                              "Another process is already running.\nPlease stop it first.")
        return False
    
    try:
        # Build command
        command = [sys.executable, executable] + args
        
        # Add debug flag if requested
        if run_with_debug and 'debug_mode' in globals() and debug_mode:
            command.append("debug")
        
        # Show command being executed
        log_message(f"Command: {' '.join(command)}")
        
        # Create the process
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,  # Line buffered
            universal_newlines=True,  # Return strings from stdout/stderr
            env=os.environ.copy()  # Copy current environment
        )
        
        # Store reference to the process
        running_process = process
        
        # Start log thread
        log_thread = threading.Thread(
            target=log_process, 
            args=(process, finish_message),
            daemon=True
        )
        log_thread.start()
        
        return True
    except Exception as e:
        log_message(f"Error starting process: {e}", level="ERROR")
        messagebox.showerror("Error", f"Failed to start process:\n{e}")
        return False

def diagnostic(ignore=None):
    """Run the diagnostic process"""
    log_message("Starting diagnostic...")
    update_status("Running diagnostic...")

    # Build arguments
    arguments = []
    
    # Add camera selection if not default
    selected_camera = camera.get()
    if selected_camera != OPTIONS[0]:
        if selected_camera == "Pi Camera" and is_raspberry_pi():
            arguments.append("cap=picam")
        else:
            cap_index = OPTIONS.index(selected_camera) - 1
            arguments.append("cap=" + str(cap_index))
    
    # Add resolution if not default
    selected_resolution = resolution.get()
    if selected_resolution != resolution_options[0]:
        width, height = selected_resolution.split(" x ")
        arguments.append(f"width={width}")
        arguments.append(f"height={height}")
    
    # Add FPS if not default
    selected_fps = fps.get()
    if selected_fps != fps_options[0]:
        arguments.append(f"fps={selected_fps}")
    
    # Add calibration if selected
    if calibration_mode.get() == CALIBRATION_OPTIONS[-1]:
        arguments.append("calibrate")
    
    # Start the process
    start_process("diagnostic.py", arguments, "Diagnostic finished.", run_with_debug=True)

def board_calibration(ignore=None):
    """Run the board calibration process"""
    # Check if calibration is needed
    if calibration_mode.get() == CALIBRATION_OPTIONS[-1]:
        messagebox.showinfo(
            "Board Calibration Not Required",
            "Calibration is not necessary for this mode. "
            "You can proceed directly without calibration."
        )
        return

    log_message("Starting board calibration...")
    update_status("Running board calibration...")
    
    # Build arguments
    arguments = ["show-info"]
    
    # Add camera selection if not default
    selected_camera = camera.get()
    if selected_camera != OPTIONS[0]:
        if selected_camera == "Pi Camera" and is_raspberry_pi():
            arguments.append("cap=picam")
        else:
            cap_index = OPTIONS.index(selected_camera) - 1
            arguments.append("cap=" + str(cap_index))
    
    # Add resolution if not default
    selected_resolution = resolution.get()
    if selected_resolution != resolution_options[0]:
        width, height = selected_resolution.split(" x ")
        arguments.append(f"width={width}")
        arguments.append(f"height={height}")
    
    # Add FPS if not default
    selected_fps = fps.get()
    if selected_fps != fps_options[0]:
        arguments.append(f"fps={selected_fps}")
    
    # Add ML mode if selected
    if calibration_mode.get() == CALIBRATION_OPTIONS[1]:
        arguments.append("ml")
    
    # Start the process
    start_process("board_calibration.py", arguments, "Board calibration finished.", run_with_debug=True)

def start_game(ignore=None):
    """Start the main game process"""
    global token
    log_message("Starting game...")
    update_status("Starting chess game...")
    
    # Check if calibration has been done
    if not os.path.exists('constants.bin'):
        result = messagebox.askyesno(
            "Calibration Required",
            "No calibration data found. Would you like to run calibration first?"
        )
        if result:
            board_calibration()
            return
    
    # Build arguments
    arguments = []
    
    # Add options based on checkboxes
    if no_template.get():
        arguments.append("no-template")
    if make_opponent.get():
        arguments.append("make-opponent")
    if comment_me.get():
        arguments.append("comment-me")
    if comment_opponent.get():
        arguments.append("comment-opponent")
    if drag_drop.get():
        arguments.append("drag")
    
    # Add token if available
    if token:
        arguments.append("token=" + token)
        promotion_menu.configure(state="normal")
        promotion.set(PROMOTION_OPTIONS[0])

    # Add delay
    arguments.append("delay=" + str(values.index(default_value.get())))

    # Add camera selection if not default
    selected_camera = camera.get()
    if selected_camera != OPTIONS[0]:
        if selected_camera == "Pi Camera" and is_raspberry_pi():
            arguments.append("cap=picam")
        else:
            cap_index = OPTIONS.index(selected_camera) - 1
            arguments.append("cap=" + str(cap_index))
    
    # Add resolution if not default
    selected_resolution = resolution.get()
    if selected_resolution != resolution_options[0]:
        width, height = selected_resolution.split(" x ")
        arguments.append(f"width={width}")
        arguments.append(f"height={height}")
    
    # Add FPS if not default
    selected_fps = fps.get()
    if selected_fps != fps_options[0]:
        arguments.append(f"fps={selected_fps}")
    
    # Add voice if not default
    selected_voice = voice.get()
    if selected_voice != VOICE_OPTIONS[0]:
        voice_index = VOICE_OPTIONS.index(selected_voice) - 1
        arguments.append("voice=" + str(voice_index))
        
        # Determine language from voice name
        language_name = "English"
        languages = ["English", "German", "Russian", "Turkish", "Italian", "French"]
        codes = ["en_", "de_", "ru_", "tr_", "it_", "fr_"]
        for l, c in zip(languages, codes):
            if (l in selected_voice) or (l.lower() in selected_voice) or (c in selected_voice):
                language_name = l
                break
        arguments.append("lang=" + language_name)

    # Add calibration if selected
    if calibration_mode.get() == CALIBRATION_OPTIONS[-1]:
        arguments.append("calibrate")

    # Check for espeak on Linux systems if speech is enabled
    if (comment_me.get() or comment_opponent.get()) and platform.system() == "Linux":
        try:
            result = subprocess.run(['which', 'espeak'], capture_output=True, text=True)
            if result.returncode != 0:
                log_message("Warning: espeak is not installed but speech is enabled", level="WARNING")
                result = messagebox.askyesno(
                    "Text-to-Speech Warning",
                    "The espeak text-to-speech engine is not installed, but speech features are enabled.\n\n"
                    "Would you like to continue anyway? (Speech will not work)"
                )
                if not result:
                    return
        except Exception:
            # Ignore errors checking for espeak
            pass
    
    # Start the process
    start_process("main.py", arguments, "Game finished.", run_with_debug=True)

def save_settings():
    """Save GUI settings to file with error handling"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(settings_file), exist_ok=True)
        
        # Create a backup of the existing file if it exists
        if os.path.exists(settings_file):
            backup_file(settings_file)
        
        # Get values from fields
        settings_data = [field.get() for field in fields] + [token]
        
        # Save to file
        with open(settings_file, 'wb') as outfile:
            pickle.dump(settings_data, outfile)
            
        log_message("Settings saved successfully", level="INFO")
        return True
    except Exception as e:
        log_message(f"Error saving settings: {e}", level="ERROR")
        traceback.print_exc()
        return False

def load_settings():
    """Load GUI settings from file with error handling"""
    global token
    
    if os.path.exists(settings_file):
        try:
            # Create a backup before loading (in case loading corrupts the file)
            backup_file(settings_file)
            
            # Load settings
            with open(settings_file, 'rb') as infile:
                variables = pickle.load(infile)
            
            # Validate the data
            if not isinstance(variables, list):
                raise ValueError("Invalid settings format")
            
            token = variables[-1] if len(variables) > len(fields) else ""
            
            # Set voice if available
            if len(variables) > 10 and variables[-2] in VOICE_OPTIONS:
                voice.set(variables[-2])
            
            # Set camera if available
            if len(variables) > 9 and variables[-3] in OPTIONS:
                camera.set(variables[-3])
            
            # Set other options
            for i in range(min(len(variables)-1, len(fields))):
                try:
                    fields[i].set(variables[i])
                except Exception as e:
                    log_message(f"Error setting field {i}: {e}", level="WARNING")
            
            log_message("Settings loaded successfully", level="INFO")
            return True
            
        except Exception as e:
            log_message(f"Error loading settings: {e}", level="ERROR")
            traceback.print_exc()
            
            # If failed to load, try to restore from backup
            backup_path = f"{settings_file}.bak"
            if os.path.exists(backup_path):
                try:
                    log_message("Attempting to restore settings from backup", level="INFO")
                    with open(backup_path, 'rb') as infile:
                        variables = pickle.load(infile)
                    

                    token = variables[-1] if len(variables) > len(fields) else ""
                    
                    # Set voice if available
                    if len(variables) > 10 and variables[-2] in VOICE_OPTIONS:
                        voice.set(variables[-2])
                    
                    # Set camera if available
                    if len(variables) > 9 and variables[-3] in OPTIONS:
                        camera.set(variables[-3])
                    
                    # Set other options
                    for i in range(min(len(variables)-1, len(fields))):
                        try:
                            fields[i].set(variables[i])
                        except Exception as e:
                            log_message(f"Error setting field {i}: {e}", level="WARNING")
                    
                    log_message("Settings restored from backup", level="INFO")
                    return True
                except Exception as backup_error:
                    log_message(f"Failed to restore from backup: {backup_error}", level="ERROR")
            
            return False
    return False

def save_promotion(*args):
    """Save promotion piece selection with error handling"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(promotion_file), exist_ok=True)
        
        # Save to file
        with open(promotion_file, 'wb') as outfile:
            pickle.dump(promotion.get(), outfile)
            
        log_message("Promotion setting saved", level="INFO", to_console=False)
        return True
    except Exception as e:
        log_message(f"Error saving promotion setting: {e}", level="ERROR")
        return False

def detect_cameras():
    """Detect available cameras with error handling"""
    cameras = ["Default"]
    
    try:
        platform_name = platform.system()
        
        if platform_name == "Darwin":  # macOS
            try:
                cmd = 'system_profiler SPCameraDataType | grep "^    [^ ]" | sed "s/    //" | sed "s/://"'
                result = subprocess.check_output(cmd, shell=True)
                result = result.decode()
                cameras.extend([r for r in result.split("\n") if r])
                log_message(f"Detected {len(cameras)-1} camera(s) on macOS")
            except Exception as e:
                log_message(f"Error detecting macOS cameras: {e}", level="WARNING")
                
        elif platform_name == "Linux":
            # Add Pi Camera if on Raspberry Pi
            if is_raspberry_pi():
                cameras.append("Pi Camera")
                log_message("Raspberry Pi camera available")
                
            # Add other video devices
            try:
                cmd = 'for I in /sys/class/video4linux/*; do cat $I/name 2>/dev/null || echo ""; done'
                result = subprocess.check_output(cmd, shell=True)
                result = result.decode()
                linux_cameras = [r for r in result.split("\n") if r]
                cameras.extend(linux_cameras)
                log_message(f"Detected {len(linux_cameras)} additional camera(s) on Linux")
            except Exception as e:
                log_message(f"Error detecting Linux cameras: {e}", level="WARNING")
                
        else:  # Windows
            try:
                from pygrabber.dshow_graph import FilterGraph
                win_cameras = FilterGraph().get_input_devices()
                cameras.extend(win_cameras)
                log_message(f"Detected {len(win_cameras)} camera(s) on Windows")
            except ImportError:
                log_message("pygrabber not available - cannot list Windows cameras", level="WARNING")
            except Exception as e:
                log_message(f"Error detecting Windows cameras: {e}", level="WARNING")
    
    except Exception as e:
        log_message(f"Camera detection error: {e}", level="ERROR")
    
    # If no cameras were detected (except Default)
    if len(cameras) == 1:
        log_message("No cameras detected", level="WARNING")
        
        # Add Pi Camera as fallback on Raspberry Pi
        if is_raspberry_pi():
            cameras.append("Pi Camera")
            log_message("Added Pi Camera as fallback option")
    
    return cameras

def detect_voices():
    """Detect available voices with error handling"""
    voices = ["Default"]
    
    try:
        platform_name = platform.system()
        
        if platform_name == "Darwin":  # macOS
            try:
                result = subprocess.run(['say', '-v', '?'], stdout=subprocess.PIPE)
                output = result.stdout.decode('utf-8')
                for line in output.splitlines():
                    if line:
                        voice_info = line.split()
                        voices.append(f'{voice_info[0]} {voice_info[1]}')
                log_message(f"Detected {len(voices)-1} voice(s) on macOS")
            except Exception as e:
                log_message(f"Error detecting macOS voices: {e}", level="WARNING")
        else:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                for v in engine.getProperty('voices'):
                    voices.append(v.name)
                log_message(f"Detected {len(voices)-1} voice(s) using pyttsx3")
            except Exception as e:
                log_message(f"Error initializing text-to-speech: {e}", level="WARNING")
                
                # Add default voices for fallback
                if is_raspberry_pi():
                    voices.extend(["English (US)", "English (UK)"])
                    log_message("Added default voices as fallback")
    
    except Exception as e:
        log_message(f"Voice detection error: {e}", level="ERROR")
    
    return voices

def create_gui():
    """Create the main GUI components"""
    global window, menu_bar, logs_text, status_label, fields
    global no_template, make_opponent, drag_drop, comment_me, comment_opponent
    global camera, resolution, fps, calibration_mode, voice, promotion
    global OPTIONS, VOICE_OPTIONS, CALIBRATION_OPTIONS, PROMOTION_OPTIONS
    global resolution_options, fps_options, default_value, values, promotion_menu
    
    # Create the main window with improved appearance
    window = tk.Tk()
    window.title(f"Chess Robot v{app_version}" + (" - Raspberry Pi Edition" if is_raspberry_pi() else ""))
    
    # Set window icon if available
    try:
        if os.path.exists("icon.png"):
            icon = tk.PhotoImage(file="icon.png")
            window.iconphoto(True, icon)
    except Exception:
        pass
    
    # Create menu bar
    menu_bar = tk.Menu(window)
    
    # File menu
    file_menu = tk.Menu(menu_bar, tearoff=False)
    file_menu.add_command(label="Save Settings", command=save_settings)
    file_menu.add_command(label="Reset Settings", command=reset_settings)
    file_menu.add_separator()
    file_menu.add_command(label="Export Logs", command=export_logs)
    file_menu.add_command(label="Clear Logs", command=clear_logs)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=on_closing)
    menu_bar.add_cascade(label="File", menu=file_menu)
    
    # Connection menu
    connection = tk.Menu(menu_bar, tearoff=False)
    connection.add_command(label="Lichess", command=lichess)
    menu_bar.add_cascade(label="Connection", menu=connection)
    
    # Tools menu
    tools_menu = tk.Menu(menu_bar, tearoff=False)
    tools_menu.add_command(label="System Information", command=lambda: check_resources())
    tools_menu.add_command(label="Backup Calibration", command=backup_calibration_data)
    menu_bar.add_cascade(label="Tools", menu=tools_menu)
    
    # Help menu
    help_menu = tk.Menu(menu_bar, tearoff=False)
    help_menu.add_command(label="Help...", command=show_help_dialog)
    help_menu.add_command(label="About...", command=show_about_dialog)
    menu_bar.add_cascade(label="Help", menu=help_menu)
    
    window.config(menu=menu_bar)

    # Create variables
    no_template = tk.IntVar()
    make_opponent = tk.IntVar()
    drag_drop = tk.IntVar()
    comment_me = tk.IntVar()
    comment_opponent = tk.IntVar()
    
    # Style configuration for ttk widgets
    style = ttk.Style()
    style.configure("TButton", padding=6, relief="flat")
    style.configure("TLabel", padding=2)
    style.configure("TCheckbutton", padding=2)
    style.configure("TFrame", padding=5)
    
    # Main frame to hold all content with padding
    main_frame = ttk.Frame(window, padding="10 10 10 10")
    main_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
    
    # Detect cameras
    OPTIONS = detect_cameras()

    # Detect voices
    VOICE_OPTIONS = detect_voices()
    
    # Create camera selection section
    camera_section = ttk.LabelFrame(main_frame, text="Camera Settings", padding=5)
    camera_section.grid(row=0, column=0, sticky="ew", padx=5, pady=5, columnspan=2)
    
    # Set up camera selection
    camera = tk.StringVar()
    camera.set(OPTIONS[0])
    
    # Camera dropdown
    camera_frame = ttk.Frame(camera_section)
    camera_frame.grid(row=0, column=0, sticky="w", padx=5, pady=2)
    
    camera_label = ttk.Label(camera_frame, text='Camera:')
    camera_label.grid(column=0, row=0, sticky=tk.W)
    
    camera_menu = ttk.Combobox(camera_frame, textvariable=camera, values=OPTIONS, state="readonly")
    camera_menu.config(width=max(len(option) for option in OPTIONS))
    camera_menu.grid(column=1, row=0, sticky=tk.W, padx=5)

    # Determine which resolution and FPS options to use based on platform
    if is_raspberry_pi():
        resolution_options = PI_RESOLUTION_OPTIONS
        fps_options = PI_FPS_OPTIONS
    else:
        resolution_options = STD_RESOLUTION_OPTIONS
        fps_options = STD_FPS_OPTIONS

    # Resolution selection
    resolution_frame = ttk.Frame(camera_section)
    resolution_frame.grid(row=1, column=0, sticky="w", padx=5, pady=2)
    
    resolution = tk.StringVar()
    resolution.set(resolution_options[0])
    
    resolution_label = ttk.Label(resolution_frame, text='Resolution:')
    resolution_label.grid(column=0, row=0, sticky=tk.W)
    
    resolution_menu = ttk.Combobox(
        resolution_frame,
        textvariable=resolution,
        values=resolution_options,
        state="readonly"
    )
    resolution_menu.config(width=max(len(option) for option in resolution_options))
    resolution_menu.grid(column=1, row=0, sticky=tk.W, padx=5)

    # FPS selection
    fps_frame = ttk.Frame(camera_section)
    fps_frame.grid(row=2, column=0, sticky="w", padx=5, pady=2)
    
    fps = tk.StringVar()
    fps.set(fps_options[0])
    
    fps_label = ttk.Label(fps_frame, text='FPS:')
    fps_label.grid(column=0, row=0, sticky=tk.W)
    
    fps_menu = ttk.Combobox(
        fps_frame,
        textvariable=fps,
        values=fps_options,
        state="readonly"
    )
    fps_menu.config(width=max(len(option) for option in fps_options))
    fps_menu.grid(column=1, row=0, sticky=tk.W, padx=5)

    # Game settings section
    game_section = ttk.LabelFrame(main_frame, text="Game Settings", padding=5)
    game_section.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

    # Calibration mode selection
    calibration_frame = ttk.Frame(game_section)
    calibration_frame.grid(row=0, column=0, sticky="w", padx=5, pady=2)
    
    calibration_mode = tk.StringVar()
    CALIBRATION_OPTIONS = ["The board is empty.", "Pieces are in their starting positions.",
                        "Just before the game starts."]
    calibration_mode.set(CALIBRATION_OPTIONS[0])
    
    calibration_label = ttk.Label(calibration_frame, text='Calibration Mode:')
    calibration_label.grid(column=0, row=0, sticky=tk.W)
    
    calibration_menu = ttk.Combobox(
        calibration_frame,
        textvariable=calibration_mode,
        values=CALIBRATION_OPTIONS,
        state="readonly"
    )
    calibration_menu.config(width=max(len(option) for option in CALIBRATION_OPTIONS))
    calibration_menu.grid(column=1, row=0, sticky=tk.W, padx=5)

    # Voice selection
    voice_frame = ttk.Frame(game_section)
    voice_frame.grid(row=1, column=0, sticky="w", padx=5, pady=2)
    
    voice = tk.StringVar()
    voice.set(VOICE_OPTIONS[0])
    
    voice_label = ttk.Label(voice_frame, text='Voice:')
    voice_label.grid(column=0, row=0, sticky=tk.W)
    
    voice_menu = ttk.Combobox(
        voice_frame,
        textvariable=voice,
        values=VOICE_OPTIONS,
        state="readonly"
    )
    voice_menu.config(width=max(len(option) for option in VOICE_OPTIONS))
    voice_menu.grid(column=1, row=0, sticky=tk.W, padx=5)

    # Promotion piece selection
    promotion_frame = ttk.Frame(game_section)
    promotion_frame.grid(row=2, column=0, sticky="w", padx=5, pady=2)
    
    promotion = tk.StringVar()
    promotion.trace("w", save_promotion)
    PROMOTION_OPTIONS = ["Queen", "Knight", "Rook", "Bishop"]
    promotion.set(PROMOTION_OPTIONS[0])
    
    promotion_label = ttk.Label(promotion_frame, text='Promotion Piece:')
    promotion_label.grid(column=0, row=0, sticky=tk.W)
    
    promotion_menu = ttk.Combobox(
        promotion_frame,
        textvariable=promotion,
        values=PROMOTION_OPTIONS,
        state="disabled"
    )
    promotion_menu.config(width=max(len(option) for option in PROMOTION_OPTIONS))
    promotion_menu.grid(column=1, row=0, sticky=tk.W, padx=5)

    # Game start delay selector
    delay_frame = ttk.Frame(game_section)
    delay_frame.grid(row=3, column=0, sticky="w", padx=5, pady=2)
    
    delay_label = ttk.Label(delay_frame, text='Game Start Delay:')
    delay_label.grid(column=0, row=0, sticky=tk.W)
    
    values = ["Do not delay game start.", "1 second delayed game start."] + [
        str(i) + " seconds delayed game start." for i in range(2, 6)
    ]
    default_value = tk.StringVar()
    default_value.set(values[-1])
    
    delay_menu = ttk.Combobox(
        delay_frame,
        textvariable=default_value,
        values=values,
        state="readonly"
    )
    delay_menu.config(width=max(len(value) for value in values))
    delay_menu.grid(column=1, row=0, sticky=tk.W, padx=5)

    # Options section
    options_section = ttk.LabelFrame(main_frame, text="Options", padding=5)
    options_section.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

    # Create checkboxes with improved layout
    c = ttk.Checkbutton(
        options_section,
        text="Auto-detect chess board in online game",
        variable=no_template
    )
    c.grid(row=0, column=0, sticky="w", padx=5, pady=3)

    c1 = ttk.Checkbutton(
        options_section,
        text="Make moves of opponent too",
        variable=make_opponent
    )
    c1.grid(row=1, column=0, sticky="w", padx=5, pady=3)

    c2 = ttk.Checkbutton(
        options_section,
        text="Make moves by drag and drop",
        variable=drag_drop
    )
    c2.grid(row=2, column=0, sticky="w", padx=5, pady=3)

    c3 = ttk.Checkbutton(
        options_section,
        text="Speak my moves",
        variable=comment_me
    )
    c3.grid(row=3, column=0, sticky="w", padx=5, pady=3)

    c4 = ttk.Checkbutton(
        options_section,
        text="Speak opponent's moves",
        variable=comment_opponent
    )
    c4.grid(row=4, column=0, sticky="w", padx=5, pady=3)

    # Create control buttons
    create_buttons()
    button_frame.grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=10)

    # Create log text area with improved styling
    text_frame = ttk.Frame(main_frame)
    text_frame.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
    
    logs_label = ttk.Label(text_frame, text="Logs:")
    logs_label.pack(anchor="w")
    
    log_frame = ttk.Frame(text_frame)
    log_frame.pack(fill="both", expand=True)
    
    scroll_bar = ttk.Scrollbar(log_frame)
    logs_text = tk.Text(
        log_frame,
        height=10,
        width=80,
        font=("Consolas", 9),
        wrap="word",
        yscrollcommand=scroll_bar.set
    )
    scroll_bar.config(command=logs_text.yview)
    scroll_bar.pack(side=tk.RIGHT, fill=tk.Y)
    logs_text.pack(side="left", fill="both", expand=True)

    # Add context menu to logs
    logs_context_menu = tk.Menu(logs_text, tearoff=0)
    logs_context_menu.add_command(label="Copy", command=lambda: logs_text.event_generate("<<Copy>>"))
    logs_context_menu.add_command(label="Select All", command=lambda: logs_text.tag_add("sel", "1.0", "end"))
    logs_context_menu.add_separator()
    logs_context_menu.add_command(label="Clear Logs", command=clear_logs)
    logs_context_menu.add_command(label="Export Logs", command=export_logs)
    
    def show_logs_context_menu(event):
        logs_context_menu.post(event.x_root, event.y_root)
    
    logs_text.bind("<Button-3>", show_logs_context_menu)  # Right-click
    
    # Create status bar
    status_label, system_info = create_status_bar()

    # Set up window to be resizable
    window.rowconfigure(0, weight=1)
    window.columnconfigure(0, weight=1)
    main_frame.rowconfigure(3, weight=1)
    main_frame.columnconfigure(0, weight=1)
    main_frame.columnconfigure(1, weight=1)
    
    # List of fields for saving settings
    fields = [no_template, make_opponent, comment_me, comment_opponent, calibration_mode, resolution, fps, drag_drop,
            default_value, camera, voice]
    
    # Log initial messages
    for log in logs:
        logs_text.insert(tk.END, log + "\n")
        
    # Add platform-specific information
    if is_raspberry_pi():
        log_message("Running on Raspberry Pi - optimized settings applied")
        log_message("Make sure your Pi Camera is properly connected and enabled")
        log_message("For best performance, use lower resolution options on Pi")
        
        # Check for required packages
        success, missing = check_dependencies()
        if not success:
            log_message(f"Missing dependencies: {', '.join(missing)}", level="WARNING")
            log_message("Install with: pip install " + " ".join(missing), level="WARNING")
    
    log_message("Ready to start", level="INFO")
    
    # Set up window close handler
    window.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Set window size and position
    window.geometry("800x600")
    window.update_idletasks()
    width = window.winfo_width()
    height = window.winfo_height()
    x = (window.winfo_screenwidth() // 2) - (width // 2)
    y = (window.winfo_screenheight() // 2) - (height // 2)
    window.geometry(f'{width}x{height}+{x}+{y}')
    
    # Set minimum window size
    window.minsize(width=600, height=500)
    
    return window

# Initialize and start the application
if __name__ == "__main__":
    try:
        # Log startup information
        log_message(f"Starting Chess Robot GUI v{app_version}", level="INFO")
        log_message(f"Config directory: {config_directory}", level="INFO")
        
        # Create main window and GUI components
        window = create_gui()
        
        # Load saved settings
        load_settings()
        
        # Start the main event loop
        window.mainloop()
    except Exception as e:
        # Handle startup errors
        print(f"Error starting application: {e}")
        traceback.print_exc()
        
        # Try to show error dialog
        try:
            messagebox.showerror(
                "Application Error",
                f"Failed to start application:\n\n{e}\n\nSee log file for details."
            )
        except:
            pass
