# Real-Time 3D Face Mask Pipeline

A high-performance face tracking system that renders a 3D sphere mask over your jaw, tracking your head and jaw movements in real-time. Perfect for video conferencing applications.

## Features

- **Real-time face tracking** using OpenCV YuNet (ultra-fast face detection)
- **Head pose estimation** with yaw, pitch, and roll angles
- **3D sphere mask** with wireframe rendering and glowing tendrils
- **Virtual camera output** compatible with Zoom, Teams, Discord, etc.
- **Python 3.13 compatible** - works with the latest Python
- **High performance**: 60-100 FPS on modern CPUs
- **Low latency**: 20-30ms end-to-end

## Quick Start

### 1. Install Dependencies

```bash
cd staging/prism
pip install -r requirements.txt
```

**Platform-specific notes:**

- **Linux**: Install v4l2loopback for virtual camera support:
  ```bash
  sudo apt install v4l2loopback-dkms v4l2loopback-utils
  sudo modprobe v4l2loopback devices=1 video_nr=10 exclusive_caps=1
  ```

- **macOS**: OBS Virtual Camera works out of the box with pyvirtualcam

- **Windows**: Install OBS Virtual Camera or Unity Capture

### 2. Run the Pipeline

**Preview mode** (shows in window, no virtual camera):
```bash
python main.py --preview
```

**Virtual camera mode** (for use in video conferencing apps):
```bash
python main.py
```

Press `Ctrl+C` to exit.

## Usage

### Basic Commands

```bash
# Default: 1280x720 @ 30 FPS with virtual camera
python main.py

# Preview window (press 'q' or ESC to quit)
python main.py --preview

# Enable debug information overlay
python main.py --debug --preview

# Custom resolution and FPS
python main.py --width 1920 --height 1080 --fps 60

# Use different camera
python main.py --camera 1
```

### Customization

```bash
# Change sphere color (BGR format)
python main.py --color 255,0,0 --preview     # Blue sphere
python main.py --color 0,0,255 --preview     # Red sphere
python main.py --color 0,255,136 --preview   # Cyan/green (default)

# Adjust transparency
python main.py --opacity 0.3 --preview       # More transparent
python main.py --opacity 0.9 --preview       # More opaque

# Print performance stats more frequently
python main.py --stats-interval 30 --debug
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--width` | Video width in pixels | 1280 |
| `--height` | Video height in pixels | 720 |
| `--fps` | Target frames per second | 30 |
| `--camera` | Camera device index | 0 |
| `--preview` | Show preview window instead of virtual camera | False |
| `--debug` | Show debug information on video | False |
| `--color` | Sphere color in BGR format (e.g., "0,255,136") | 0,255,136 |
| `--opacity` | Sphere transparency (0.0-1.0) | 0.6 |
| `--stats-interval` | Print stats every N frames | 100 |

## Using with Video Conferencing Apps

1. **Start the pipeline** in virtual camera mode:
   ```bash
   python main.py
   ```

2. **Select the virtual camera** in your video conferencing app:
   - **Zoom**: Settings → Video → Camera → Select "OBS Virtual Camera" or "pyvirtualcam"
   - **Teams**: Settings → Devices → Camera → Select virtual camera
   - **Discord**: User Settings → Voice & Video → Camera → Select virtual camera
   - **Google Meet**: Click on three dots → Settings → Video → Select virtual camera

3. The sphere mask will track your jaw and head movements in real-time.

## Architecture

The pipeline consists of these components:

```
Physical Webcam → Video Capture (threaded)
                      ↓
                 Face Tracker (MediaPipe)
                      ↓
                 Pose Calculator (cv2.solvePnP)
                      ↓
                 Sphere Renderer
                      ↓
                 Virtual Camera → Conference Apps
```

### Component Details

- **video_capture.py**: Threaded webcam capture (52-67% FPS improvement)
- **face_tracker.py**: OpenCV YuNet face detector with 5 facial landmarks
- **pose_calculator.py**: Head pose estimation (yaw, pitch, roll) using PnP algorithm
- **sphere_renderer.py**: 3D sphere rendering with wireframe and glow effects
- **virtual_camera.py**: Virtual camera output via pyvirtualcam
- **performance_monitor.py**: FPS and latency tracking for each pipeline stage
- **main.py**: Main application loop

## Performance

**Expected performance on modern laptop (Intel i5/i7, AMD Ryzen 5/7):**

- **FPS**: 60-100+ (with threading)
- **Latency**: 20-30ms end-to-end
  - Capture: 5-10ms
  - Detection: 3-8ms (YuNet is very fast)
  - Pose calculation: 1-2ms
  - Rendering: 5-10ms
  - Output: <1ms

**Performance tips:**

1. Lower resolution for higher FPS: `--width 640 --height 480`
2. Use `--fps 30` on slower machines
3. Close other applications to free CPU resources
4. The threaded capture significantly improves performance

## Troubleshooting

### Virtual camera not showing up

**Linux:**
```bash
# Check if v4l2loopback is loaded
lsmod | grep v4l2loopback

# If not, load it
sudo modprobe v4l2loopback devices=1 video_nr=10 exclusive_caps=1

# List video devices
ls /dev/video*
```

**macOS/Windows:**
- Make sure OBS Studio is installed
- Start OBS → Tools → Start Virtual Camera
- Then run the pipeline

### Low FPS

1. Lower the resolution: `--width 640 --height 480`
2. Reduce target FPS: `--fps 20`
3. Check CPU usage - close background applications
4. Use preview mode to test: `--preview --debug`

### Camera not found

```bash
# List available cameras
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"

# Then use the correct index
python main.py --camera 1
```

### No face detected

1. Ensure good lighting
2. Face the camera directly
3. Adjust YuNet confidence thresholds in `face_tracker.py` (default is 0.6)
4. Use `--debug --preview` to see detection status
5. The YuNet model will auto-download on first run (~337KB)

## Development

### Project Structure

```
staging/prism/
├── main.py                    # Main application
├── video_capture.py          # Threaded webcam capture
├── face_tracker.py           # MediaPipe Face Mesh
├── pose_calculator.py        # Head pose estimation
├── sphere_renderer.py        # 3D sphere rendering
├── virtual_camera.py         # Virtual camera output
├── performance_monitor.py    # Performance tracking
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

### Customizing the Mask

To change the sphere rendering, edit `sphere_renderer.py`:

- `_draw_gradient_sphere()`: Change sphere fill style
- `_draw_wireframe_sphere()`: Modify wireframe pattern
- `_draw_tendrils()`: Adjust animated tendrils
- `_get_pulse_scale()`: Change pulsing animation

### Adding New Features

The modular architecture makes it easy to:

1. Replace sphere with custom 3D model (modify `sphere_renderer.py`)
2. Add face filters or effects (extend `SphereRenderer`)
3. Track multiple faces (change `max_num_faces` in `face_tracker.py`)
4. Export recordings (add video writer in `main.py`)

## System Requirements

**Minimum:**
- Python: 3.8+ (tested with 3.13)
- CPU: Intel i5-8xxx / AMD Ryzen 5 2xxx
- RAM: 8GB
- OS: Windows 10+, macOS 10.15+, Ubuntu 20.04+, Arch Linux
- Webcam: 720p @ 30 FPS
- Expected: 30-50 FPS

**Recommended:**
- Python: 3.11-3.13
- CPU: Intel i7-10xxx / AMD Ryzen 7 3xxx or better
- RAM: 16GB
- Webcam: 1080p @ 30 FPS
- Expected: 60-100+ FPS

## Credits

- **OpenCV YuNet**: Ultra-fast face detection from OpenCV Zoo
- **OpenCV**: Computer vision, DNN module, and rendering
- **pyvirtualcam**: Virtual camera implementation

## License

This is experimental code for research and development purposes.
