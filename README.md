# Chess Vision Robot Integration System

A comprehensive system that combines computer vision and robotics to play chess on a real chess board. This project integrates the [Play-online-chess-with-real-chess-board](https://github.com/karayaman/Play-online-chess-with-real-chess-board) vision system with a custom 3D printer-based robotic arm to enable a physical robot to play chess against a human opponent.
## Features

- **Computer Vision**: Detects the chess board and pieces using camera input
- **Move Detection**: Automatically recognizes when a human player makes a move
- **Robotic Integration**: Controls a modified 3D printer to move chess pieces
- **Chess Engine**: Uses Stockfish for AI move generation with configurable difficulty levels
- **Speech Feedback**: Provides audio feedback and move announcements
- **Error Recovery**: Robust error handling and recovery mechanisms
- **Diagnostics**: Built-in calibration and diagnostic tools

## System Requirements

- Raspberry Pi 4 (recommended) or other Linux system
- USB or Pi Camera
- Modified 3D printer (with gripper attachment)
- Chess board and standard chess pieces
- Python 3.7+

## Dependencies

- OpenCV
- NumPy
- python-chess
- PySerial
- gTTS (for speech)
- psutil (for system monitoring)

## Project Structure

```
chess-robot-integration/
├── chess_vision_robot_integration.py   # Main integration script
├── config/                            # Configuration files
├── logs/                              # Log output directory
├── debug/                             # Debug image storage
├── Play-online-chess-with-real-chess-board/  # Vision system submodule
└── PrinterController/                 # Robot control submodule
```

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/chess-robot-integration.git
   cd chess-robot-integration
   ```

2. Clone the required submodules:
   ```bash
   git submodule add https://github.com/karayaman/Play-online-chess-with-real-chess-board.git
   git submodule add https://github.com/yourusername/PrinterController.git
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure your 3D printer and camera setup (see Configuration section)

## Configuration

The system uses a configuration file located at `config/integration_config.json` that controls various aspects of the system. Example configuration:

```json
{
  "camera": {
    "index": 0,
    "width": 1280,
    "height": 720,
    "fps": 30
  },
  "printer": {
    "port": "/dev/ttyUSB0",
    "baud_rate": 115200
  },
  "chess": {
    "engine_path": "",
    "skill_level": 10,
    "thinking_time": 1.0
  },
  "speech": {
    "enabled": true,
    "language": "en",
    "volume": 1.0,
    "rate": 150
  },
  "debug": false
}
```

## Usage

1. Run the main integration script:
   ```bash
   python chess_vision_robot_integration.py
   ```

2. Follow the menu options:
   - Initialize the system
   - Run calibration to detect the chess board
   - Play a game against the robot

## Calibration

Before playing, you need to calibrate the system so it can recognize the chess board:

1. Select option 4 from the main menu
2. Make sure the chess board is visible to the camera
3. Follow the on-screen instructions to complete calibration

## Playing a Game

1. Initialize the system (option 1)
2. Select "Play game" (option 2) 
3. Choose your color (White/Black)
4. Select difficulty level
5. Make your moves on the physical board
6. The system will detect your move and respond with the robot's move

## Troubleshooting

- **Camera not detected**: Check USB connections and permissions
- **Printer not responsive**: Verify serial port settings and connection
- **Move detection issues**: Run calibration again with better lighting
- **Mechanical problems**: Use the diagnostic options to test printer movements

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Karayaman](https://github.com/karayaman) for the chess vision system
- Chess engine based on [Stockfish](https://stockfishchess.org/)
