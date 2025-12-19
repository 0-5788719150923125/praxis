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
from utils import load_json, get_absolute_path, get_video_info, ensure_dir


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


def create_mlt_project(video_path: str, events: list, fps: float, output_path: str):
    """
    Generate Shotcut MLT XML project file matching Shotcut's structure.

    Args:
        video_path: Absolute path to source video
        events: List of event dicts with start_time and end_time
        fps: Video frame rate
        output_path: Where to save .mlt file
    """
    # Ensure video path is absolute
    video_path = get_absolute_path(video_path)

    if not os.path.exists(video_path):
        print(f"Warning: Video file not found: {video_path}")

    print(f"Generating MLT project...")
    print(f"  Source video: {video_path}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Events: {len(events)}")
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

    # Create a chain for each event (clip)
    video_name = Path(video_path).stem
    for i, event in enumerate(events):
        chain_id = f"chain{i}"
        event_id = event['event_id']
        start_time = event['start_time']
        end_time = event['end_time']

        # Create chain element
        chain = etree.SubElement(mlt, 'chain', id=chain_id, out=video_duration_tc)
        etree.SubElement(chain, 'property', name='length').text = video_duration_tc
        etree.SubElement(chain, 'property', name='eof').text = 'pause'
        etree.SubElement(chain, 'property', name='resource').text = video_path
        etree.SubElement(chain, 'property', name='mlt_service').text = 'avformat-novalidate'
        etree.SubElement(chain, 'property', name='shotcut:caption').text = video_name
        etree.SubElement(chain, 'property', name='xml').text = 'was here'

        print(f"  Event {event_id}: {format_timecode(start_time, fps)} - {format_timecode(end_time, fps)}")

    # Create video playlist with all clips placed sequentially
    playlist0 = etree.SubElement(mlt, 'playlist', id='playlist0')
    etree.SubElement(playlist0, 'property', name='shotcut:video').text = '1'
    etree.SubElement(playlist0, 'property', name='shotcut:name').text = 'V1'

    # Place clips sequentially on timeline
    timeline_position = 0.0
    for i, event in enumerate(events):
        chain_id = f"chain{i}"
        start_time = event['start_time']
        end_time = event['end_time']
        clip_duration = end_time - start_time

        # Entry references the chain and specifies which part of the source to use
        entry = etree.SubElement(playlist0, 'entry',
                                producer=chain_id,
                                **{'in': format_timecode(start_time, fps),
                                   'out': format_timecode(end_time, fps)})

        timeline_position += clip_duration

    # Calculate total timeline duration
    total_duration = sum(event['end_time'] - event['start_time'] for event in events)
    total_duration_tc = format_timecode(total_duration, fps)

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
    print(f"Total timeline duration: {total_duration_tc} ({len(events)} clips)")
    print(f"\nTo use:")
    print(f"  1. Open Shotcut: shotcut {output_path}")
    print(f"  2. Preview the clips in the timeline")
    print(f"  3. File → Export → choose format and render")


def generate_from_events_file(events_file: str, output_path: str = None):
    """
    Generate MLT project from events JSON file.

    Args:
        events_file: Path to events JSON file
        output_path: Output MLT path (optional)
    """
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
    create_mlt_project(video_path, events, fps, output_path)


def main():
    parser = argparse.ArgumentParser(description='Generate Shotcut MLT project')
    parser.add_argument('--events', required=True, help='Path to events JSON file')
    parser.add_argument('--output', help='Output MLT path (default: outputs/projects/<video>_project.mlt)')

    args = parser.parse_args()

    generate_from_events_file(args.events, args.output)


if __name__ == '__main__':
    main()
