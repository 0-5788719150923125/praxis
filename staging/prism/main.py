#!/usr/bin/env python3
"""
Real-Time 3D Face Mask Pipeline
Tracks face and jaw movement, renders 3D sphere mask, outputs to virtual camera.
"""

import argparse
import time
import sys

from video_capture import VideoCapture
from face_tracker import FaceTracker
from pose_calculator import PoseCalculator
from tetrahedron_renderer import TetrahedronRenderer
from virtual_camera import VirtualCamera, PreviewWindow
from performance_monitor import PerformanceMonitor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Real-time 3D face mask with jaw tracking'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=1920,
        help='Video width (default: 1920)'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=1080,
        help='Video height (default: 1080)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Target FPS (default: 30)'
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera index (default: 0)'
    )
    parser.add_argument(
        '--preview',
        action='store_true',
        help='Show preview window instead of virtual camera'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Show debug information on video'
    )
    parser.add_argument(
        '--color',
        type=str,
        default='0,255,136',
        help='Tetrahedron color in BGR format (default: 0,255,136 - cyan/green)'
    )
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.6,
        help='Tetrahedron opacity 0-1 (default: 0.6)'
    )
    parser.add_argument(
        '--smoothing',
        type=float,
        default=0.7,
        help='Rotation smoothing 0-1, higher=smoother but laggier (default: 0.7)'
    )
    parser.add_argument(
        '--stats-interval',
        type=int,
        default=100,
        help='Print stats every N frames (default: 100)'
    )
    parser.add_argument(
        '--process-width',
        type=int,
        default=None,
        help='Processing resolution width (default: same as output). Set lower for better performance.'
    )
    parser.add_argument(
        '--process-height',
        type=int,
        default=None,
        help='Processing resolution height (default: same as output). Set lower for better performance.'
    )

    return parser.parse_args()


def main():
    """Main application loop."""
    args = parse_args()

    # Parse color
    try:
        color = tuple(map(int, args.color.split(',')))
        if len(color) != 3:
            raise ValueError
    except:
        print(f"Error: Invalid color format '{args.color}'. Use BGR format like '0,255,136'")
        return 1

    # Determine processing resolution (defaults to output resolution)
    process_width = args.process_width if args.process_width else args.width
    process_height = args.process_height if args.process_height else args.height

    print("="*60)
    print("Real-Time 3D Tetrahedron Face Mask Pipeline")
    print("="*60)
    print(f"Output Resolution: {args.width}x{args.height} @ {args.fps} FPS")
    if process_width != args.width or process_height != args.height:
        print(f"Processing Resolution: {process_width}x{process_height} (optimized)")
    print(f"Camera: {args.camera}")
    print(f"Tetrahedron Color (BGR): {color}")
    print(f"Tetrahedron Opacity: {args.opacity}")
    print(f"Smoothing: {args.smoothing}")
    print(f"Mode: {'Preview Window' if args.preview else 'Virtual Camera'}")
    print(f"Debug: {'Enabled' if args.debug else 'Disabled'}")
    print("="*60)
    print()

    # Initialize components
    print("Initializing components...")

    capture = VideoCapture(
        src=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps
    )

    if not capture.is_opened():
        print(f"Error: Could not open camera {args.camera}")
        return 1

    tracker = FaceTracker()
    pose_calc = PoseCalculator(image_width=process_width, image_height=process_height)
    renderer = TetrahedronRenderer(color=color, opacity=args.opacity, smoothing=args.smoothing)
    monitor = PerformanceMonitor()

    # Store scale factors for coordinate mapping
    scale_x = args.width / process_width
    scale_y = args.height / process_height

    print("Components initialized.")
    print()

    # Start video capture thread
    capture.start()
    time.sleep(1)  # Allow camera to warm up

    print("Starting processing...")
    print("Press Ctrl+C to exit")
    print()

    try:
        if args.preview:
            # Preview mode - show in OpenCV window
            with PreviewWindow() as preview:
                run_pipeline(
                    capture, tracker, pose_calc, renderer, monitor,
                    output=preview, args=args,
                    process_width=process_width, process_height=process_height,
                    scale_x=scale_x, scale_y=scale_y
                )
        else:
            # Virtual camera mode
            with VirtualCamera(width=args.width, height=args.height, fps=args.fps) as vcam:
                run_pipeline(
                    capture, tracker, pose_calc, renderer, monitor,
                    output=vcam, args=args,
                    process_width=process_width, process_height=process_height,
                    scale_x=scale_x, scale_y=scale_y
                )

    except KeyboardInterrupt:
        print("\n\nShutting down...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Cleanup
        capture.stop()
        tracker.close()
        print("Cleanup complete.")

    # Print final stats
    print("\nFinal Performance Statistics:")
    monitor.print_stats(detailed=True)

    return 0


def run_pipeline(capture, tracker, pose_calc, renderer, monitor, output, args,
                 process_width, process_height, scale_x, scale_y):
    """
    Run the main processing pipeline.

    Args:
        capture: VideoCapture instance
        tracker: FaceTracker instance
        pose_calc: PoseCalculator instance
        renderer: TetrahedronRenderer instance
        monitor: PerformanceMonitor instance
        output: VirtualCamera or PreviewWindow instance
        args: Command line arguments
        process_width: Width for face detection processing
        process_height: Height for face detection processing
        scale_x: X scaling factor (output_width / process_width)
        scale_y: Y scaling factor (output_height / process_height)
    """
    import cv2
    use_dual_resolution = (scale_x != 1.0 or scale_y != 1.0)

    while True:
        # Read frame from capture thread (high resolution)
        with monitor.measure('capture'):
            frame_highres = capture.read()

        # Downscale for processing if using dual resolution
        if use_dual_resolution:
            with monitor.measure('downscale'):
                frame_process = cv2.resize(frame_highres, (process_width, process_height),
                                          interpolation=cv2.INTER_LINEAR)
        else:
            frame_process = frame_highres

        # Detect face landmarks on low-res frame
        with monitor.measure('detection'):
            face_data, image_shape = tracker.detect(frame_process)

        # Calculate pose if face detected
        pose_data = None
        jaw_points = None
        jaw_width = 0

        if face_data is not None:
            # Get jaw points (on low-res frame)
            with monitor.measure('landmarks'):
                jaw_points_lowres = tracker.get_jaw_points(face_data, image_shape)
                jaw_width_lowres = tracker.calculate_jaw_width(jaw_points_lowres)

            # Calculate head pose (on low-res frame)
            with monitor.measure('pose'):
                points_2d, points_3d = tracker.get_pose_points(face_data, image_shape)
                rotation_matrix, (yaw, pitch, roll) = pose_calc.calculate_pose(points_2d, points_3d)
                pose_data = (rotation_matrix, (yaw, pitch, roll))

            # Scale jaw points and width to high-res coordinates
            if use_dual_resolution:
                jaw_points = {
                    'chin': (int(jaw_points_lowres['chin'][0] * scale_x),
                            int(jaw_points_lowres['chin'][1] * scale_y),
                            jaw_points_lowres['chin'][2]),
                    'left_jaw': (int(jaw_points_lowres['left_jaw'][0] * scale_x),
                                int(jaw_points_lowres['left_jaw'][1] * scale_y),
                                jaw_points_lowres['left_jaw'][2]),
                    'right_jaw': (int(jaw_points_lowres['right_jaw'][0] * scale_x),
                                 int(jaw_points_lowres['right_jaw'][1] * scale_y),
                                 jaw_points_lowres['right_jaw'][2])
                }
                jaw_width = jaw_width_lowres * scale_x
            else:
                jaw_points = jaw_points_lowres
                jaw_width = jaw_width_lowres

        # Render tetrahedron mask on high-res frame
        with monitor.measure('rendering'):
            output_frame = renderer.render(frame_highres, jaw_points, jaw_width, pose_data)

        # Add debug info if requested
        if args.debug:
            fps = monitor.get_fps()
            output_frame = renderer.draw_debug_info(
                output_frame, jaw_points, jaw_width, pose_data, fps
            )

        # Send to output (virtual camera or preview)
        with monitor.measure('output'):
            if isinstance(output, VirtualCamera):
                output.send(output_frame)
                output.sleep_until_next_frame()
            else:  # PreviewWindow
                key = output.show(output_frame)
                if key == ord('q') or key == 27 or output.is_closed():  # q or ESC
                    break

        # Record frame processed
        monitor.record_frame()

        # Print stats periodically
        if monitor.should_print_stats(args.stats_interval):
            monitor.print_stats(detailed=args.debug)


if __name__ == '__main__':
    sys.exit(main())
