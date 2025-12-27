#!/usr/bin/env python3
"""
Generate Shotcut MLT project file from detected events.

Usage:
    python src/generate_mlt.py --events outputs/events/my_video_events.json
    python src/generate_mlt.py --events outputs/events/my_video_events.json --output project.mlt
"""

import os
import argparse
from pathlib import Path
from lxml import etree
from utils import load_json, get_absolute_path, get_video_info, ensure_dir, load_config


def format_timecode(seconds: float, fps: float) -> str:
    """Convert seconds to Shotcut timecode format HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    # Calculate milliseconds based on frame position
    fractional = seconds - int(seconds)
    frames = int(fractional * fps)
    milliseconds = int((frames / fps) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def create_mlt_project(video_path: str, events: list, fps: float, output_path: str, marker_buffer: float = 2.0):
    """
    Generate Shotcut MLT XML project file with timeline cuts at prediction points.

    Args:
        video_path: Absolute path to source video
        events: List of event dicts with start_time and end_time
        fps: Video frame rate
        output_path: Where to save .mlt file
        marker_buffer: Seconds before each event to place cut (default: 2.0)
    """
    # Ensure video path is absolute
    video_path = get_absolute_path(video_path)

    if not os.path.exists(video_path):
        print(f"Warning: Video file not found: {video_path}")

    print(f"Generating MLT project with timeline cuts...")
    print(f"  Source video: {video_path}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Events: {len(events)}")
    print(f"  Marker buffer: {marker_buffer}s")
    print()

    # Get video info for metadata
    video_info = get_video_info(video_path)
    video_duration = video_info['duration']
    video_duration_tc = format_timecode(video_duration, fps)

    # Create MLT XML structure
    mlt = etree.Element('mlt',
                        LC_NUMERIC="C",
                        version='7.34.1',
                        title='Shotcut version 25.08.16',
                        producer='main_bin')

    # Add profile matching video specs
    fps_num = int(fps * 1000)
    fps_den = 1000 if fps < 60 else 1001  # Use 1001 for 60fps (59.94)
    if fps > 59:
        fps_num = 60000
        fps_den = 1001

    profile = etree.SubElement(mlt, 'profile',
                               description='automatic',
                               width=str(video_info['width']),
                               height=str(video_info['height']),
                               progressive='1',
                               sample_aspect_num='1',
                               sample_aspect_den='1',
                               display_aspect_num='16',
                               display_aspect_den='9',
                               frame_rate_num=str(fps_num),
                               frame_rate_den=str(fps_den),
                               colorspace='709')

    # Main bin playlist (required by Shotcut)
    main_bin = etree.SubElement(mlt, 'playlist', id='main_bin')
    etree.SubElement(main_bin, 'property', name='xml_retain').text = '1'

    # Black background producer
    black = etree.SubElement(mlt, 'producer', id='black',
                             **{'in': '00:00:00.000', 'out': video_duration_tc})
    etree.SubElement(black, 'property', name='length').text = video_duration_tc
    etree.SubElement(black, 'property', name='eof').text = 'pause'
    etree.SubElement(black, 'property', name='resource').text = '0'
    etree.SubElement(black, 'property', name='aspect_ratio').text = '1'
    etree.SubElement(black, 'property', name='mlt_service').text = 'color'
    etree.SubElement(black, 'property', name='mlt_image_format').text = 'rgba'
    etree.SubElement(black, 'property', name='set.test_audio').text = '0'

    # Background playlist
    background = etree.SubElement(mlt, 'playlist', id='background')
    etree.SubElement(background, 'entry',
                     producer='black',
                     **{'in': '00:00:00.000', 'out': video_duration_tc})

    # Create a single chain for the video
    video_name = Path(video_path).stem
    chain = etree.SubElement(mlt, 'chain', id='chain0', out=video_duration_tc)
    etree.SubElement(chain, 'property', name='length').text = video_duration_tc
    etree.SubElement(chain, 'property', name='eof').text = 'pause'
    etree.SubElement(chain, 'property', name='resource').text = video_path
    etree.SubElement(chain, 'property', name='mlt_service').text = 'avformat-novalidate'
    etree.SubElement(chain, 'property', name='shotcut:caption').text = video_name
    etree.SubElement(chain, 'property', name='xml').text = 'was here'

    # Calculate cut points (marker_buffer seconds before each event)
    raw_cut_points = []
    for event in events:
        cut_time = max(0, event['start_time'] - marker_buffer)
        raw_cut_points.append(cut_time)
        print(f"  Raw cut at {format_timecode(cut_time, fps)} (event at {format_timecode(event['start_time'], fps)})")

    # Filter out cuts too close to start/end (they create zero-duration segments)
    min_cut_time = 0.1  # Don't cut in first 0.1 seconds
    max_cut_time = video_duration - 0.1  # Don't cut in last 0.1 seconds

    cut_points = []
    for cut_time in raw_cut_points:
        if min_cut_time <= cut_time <= max_cut_time:
            cut_points.append(cut_time)
        else:
            print(f"  Skipped cut at {format_timecode(cut_time, fps)} (too close to video boundary)")

    # Sort and deduplicate cut points
    cut_points = sorted(set(cut_points))

    if not cut_points:
        print("\n⚠ WARNING: No valid cut points after filtering!")
        print("  All events are either at the very start or very end of the video.")
        print("  The timeline will have no cuts.")

    # Create video playlist with segments split at cut points
    playlist0 = etree.SubElement(mlt, 'playlist', id='playlist0')
    etree.SubElement(playlist0, 'property', name='shotcut:video').text = '1'
    etree.SubElement(playlist0, 'property', name='shotcut:name').text = 'V1'

    # Create segments between cuts
    if cut_points:
        # Create segments between cuts
        segment_starts = [0.0] + cut_points
        segment_ends = cut_points + [video_duration]

        for i, (start, end) in enumerate(zip(segment_starts, segment_ends)):
            # Each segment references the same chain but with different in/out points
            entry = etree.SubElement(playlist0, 'entry',
                                    producer='chain0',
                                    **{'in': format_timecode(start, fps),
                                       'out': format_timecode(end, fps)})

        print(f"\nCreated {len(segment_starts)} segments with {len(cut_points)} cut points")
    else:
        # No cuts - just add the entire video as one segment
        entry = etree.SubElement(playlist0, 'entry',
                                producer='chain0',
                                **{'in': format_timecode(0.0, fps),
                                   'out': format_timecode(video_duration, fps)})
        print(f"\nCreated 1 segment (no cuts)")

    # Timeline duration is the full video duration
    total_duration_tc = video_duration_tc

    # Create tractor combining background and video tracks
    tractor = etree.SubElement(mlt, 'tractor',
                               id='tractor0',
                               title='Shotcut version 25.08.16',
                               **{'in': '00:00:00.000', 'out': total_duration_tc})
    etree.SubElement(tractor, 'property', name='shotcut').text = '1'
    etree.SubElement(tractor, 'property', name='shotcut:projectAudioChannels').text = '2'
    etree.SubElement(tractor, 'property', name='shotcut:projectFolder').text = '1'

    # Add tracks: background + video playlist
    etree.SubElement(tractor, 'track', producer='background')
    etree.SubElement(tractor, 'track', producer='playlist0')

    # Add transitions (audio mix + video blend)
    transition0 = etree.SubElement(tractor, 'transition', id='transition0')
    etree.SubElement(transition0, 'property', name='a_track').text = '0'
    etree.SubElement(transition0, 'property', name='b_track').text = '1'
    etree.SubElement(transition0, 'property', name='mlt_service').text = 'mix'
    etree.SubElement(transition0, 'property', name='always_active').text = '1'
    etree.SubElement(transition0, 'property', name='sum').text = '1'

    transition1 = etree.SubElement(tractor, 'transition', id='transition1')
    etree.SubElement(transition1, 'property', name='a_track').text = '0'
    etree.SubElement(transition1, 'property', name='b_track').text = '1'
    etree.SubElement(transition1, 'property', name='version').text = '0.1'
    etree.SubElement(transition1, 'property', name='mlt_service').text = 'frei0r.cairoblend'
    etree.SubElement(transition1, 'property', name='threads').text = '0'
    etree.SubElement(transition1, 'property', name='disable').text = '1'

    # Write to file
    ensure_dir(os.path.dirname(output_path))

    tree = etree.ElementTree(mlt)
    tree.write(output_path,
               pretty_print=True,
               xml_declaration=True,
               encoding='utf-8',
               standalone=False)

    print(f"\nMLT project saved to: {output_path}")
    print(f"Total timeline duration: {total_duration_tc}")
    print(f"Video split into {len(segment_starts)} segments at {len(cut_points)} cut points")
    print(f"\nTo use:")
    print(f"  1. Open Shotcut: shotcut {output_path}")
    print(f"  2. Timeline shows the full video with cuts at predicted moments")
    print(f"  3. File → Export → choose format and render")


def generate_from_events_file(events_file: str, output_path: str = None, config_path: str = 'config.yaml'):
    """
    Generate MLT project from events JSON file.

    Args:
        events_file: Path to events JSON file
        output_path: Output MLT path (optional)
        config_path: Path to config file (default: config.yaml)
    """
    # Load config
    config = load_config(config_path)
    marker_buffer = config.get('mlt', {}).get('marker_buffer', 2.0)

    # Load events
    print(f"Loading events from: {events_file}")
    data = load_json(events_file)

    video_path = data['video_path']
    video_info = data['video_info']
    events = data['events']
    fps = video_info['fps']

    if not events:
        print("No events found! Nothing to generate.")
        return

    # Determine output path
    if output_path is None:
        video_name = Path(video_path).stem
        output_path = f"outputs/projects/{video_name}_project.mlt"

    # Generate MLT
    create_mlt_project(video_path, events, fps, output_path, marker_buffer)


def main():
    parser = argparse.ArgumentParser(description='Generate Shotcut MLT project')
    parser.add_argument('--events', required=True, help='Path to events JSON file')
    parser.add_argument('--output', help='Output MLT path (default: outputs/projects/<video>_project.mlt)')

    args = parser.parse_args()

    generate_from_events_file(args.events, args.output)


if __name__ == '__main__':
    main()
