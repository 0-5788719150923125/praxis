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

    print("="*60)
    print("Real-Time 3D Tetrahedron Face Mask Pipeline")
    print("="*60)
    print(f"Resolution: {args.width}x{args.height} @ {args.fps} FPS")
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
    pose_calc = PoseCalculator(image_width=args.width, image_height=args.height)
    renderer = TetrahedronRenderer(color=color, opacity=args.opacity, smoothing=args.smoothing)
    monitor = PerformanceMonitor()

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
                    output=preview, args=args
                )
        else:
            # Virtual camera mode
            with VirtualCamera(width=args.width, height=args.height, fps=args.fps) as vcam:
                run_pipeline(
                    capture, tracker, pose_calc, renderer, monitor,
                    output=vcam, args=args
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


def run_pipeline(capture, tracker, pose_calc, renderer, monitor, output, args):
    """
    Run the main processing pipeline.

    Args:
        capture: VideoCapture instance
        tracker: FaceTracker instance
        pose_calc: PoseCalculator instance
        renderer: SphereRenderer instance
        monitor: PerformanceMonitor instance
        output: VirtualCamera or PreviewWindow instance
        args: Command line arguments
    """
    while True:
        # Read frame from capture thread
        with monitor.measure('capture'):
            frame = capture.read()

        # Detect face landmarks
        with monitor.measure('detection'):
            face_data, image_shape = tracker.detect(frame)

        # Calculate pose if face detected
        pose_data = None
        jaw_points = None
        jaw_width = 0

        if face_data is not None:
            # Get jaw points
            with monitor.measure('landmarks'):
                jaw_points = tracker.get_jaw_points(face_data, image_shape)
                jaw_width = tracker.calculate_jaw_width(jaw_points)

            # Calculate head pose
            with monitor.measure('pose'):
                points_2d, points_3d = tracker.get_pose_points(face_data, image_shape)
                rotation_matrix, (yaw, pitch, roll) = pose_calc.calculate_pose(points_2d, points_3d)
                pose_data = (rotation_matrix, (yaw, pitch, roll))

        # Render tetrahedron mask
        with monitor.measure('rendering'):
            output_frame = renderer.render(frame, jaw_points, jaw_width, pose_data)

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
