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
from naming import get_experiment_path


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


def merge_overlapping_clips(clips: list, fps: float) -> list:
    """
    Merge clips that overlap or are adjacent after applying buffers.

    This prevents duplicate content when events are close together.
    For example, if two events are 2 seconds apart but each has 1s pre-buffer
    and 2s post-buffer, their clips would overlap by 1 second. This function
    merges such clips into a single continuous clip.

    Args:
        clips: List of clip dicts with 'start', 'end', 'duration' keys
        fps: Video frame rate (for logging)

    Returns:
        List of merged clips (non-overlapping)
    """
    if not clips:
        return []

    # Sort clips by start time
    sorted_clips = sorted(clips, key=lambda c: c['start'])
    merged = []

    for current_clip in sorted_clips:
        if not merged:
            # First clip - initialize with event_ids list
            current_clip['event_ids'] = [current_clip.get('event_id', 1)]
            merged.append(current_clip)
        else:
            previous_clip = merged[-1]

            # Check for overlap: current starts before or at previous end
            if current_clip['start'] <= previous_clip['end']:
                # Overlap detected - merge clips
                original_end = previous_clip['end']
                previous_clip['end'] = max(previous_clip['end'], current_clip['end'])
                previous_clip['duration'] = previous_clip['end'] - previous_clip['start']

                # Track which events were merged
                previous_clip['event_ids'].append(current_clip.get('event_id', len(merged) + 1))

                # Log the merge
                overlap = original_end - current_clip['start']
                print(f"  ↳ Merged overlapping clips (overlap: {overlap:.2f}s) - "
                      f"Events {previous_clip['event_ids']} → "
                      f"Combined clip: {format_timecode(previous_clip['start'], fps)} - "
                      f"{format_timecode(previous_clip['end'], fps)}")
            else:
                # No overlap - keep as separate clip
                current_clip['event_ids'] = [current_clip.get('event_id', len(merged) + 1)]
                merged.append(current_clip)

    return merged


def create_mlt_project(video_path: str, events: list, fps: float, output_path: str,
                       marker_buffer: float = 2.0, post_buffer: float = 1.0, mode: str = 'cut_markers',
                       mute_audio: bool = False, add_benny_hill: bool = False, vertical_format: bool = False):
    """
    Generate Shotcut MLT XML project file with timeline cuts or extracted clips.

    Args:
        video_path: Absolute path to source video
        events: List of event dicts with start_time and end_time
        fps: Video frame rate
        output_path: Where to save .mlt file
        marker_buffer: Seconds before each event to place cut (default: 2.0)
        post_buffer: Seconds after event end for extract mode (default: 1.0)
        mode: 'cut_markers' for full video with cuts, 'extract_clips' for montage (default: 'cut_markers')
        mute_audio: Mute source video's audio track (default: False)
        add_benny_hill: Add Benny Hill theme song audio track (default: False)
        vertical_format: Use vertical 1080x1920 format with zoom (default: False)
    """
    # Ensure video path is absolute
    video_path = get_absolute_path(video_path)

    if not os.path.exists(video_path):
        print(f"Warning: Video file not found: {video_path}")

    mode_desc = "timeline cuts" if mode == 'cut_markers' else "extracted clips montage"
    print(f"Generating MLT project with {mode_desc}...")
    print(f"  Source video: {video_path}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Events: {len(events)}")
    print(f"  Mode: {mode}")
    if mode == 'cut_markers':
        print(f"  Marker buffer: {marker_buffer}s")
    else:
        print(f"  Pre-event buffer: {marker_buffer}s")
        print(f"  Post-event buffer: {post_buffer}s")
    print(f"  Source video audio: {'Muted' if mute_audio else 'Enabled'}")
    print(f"  Benny Hill theme: {'Added' if add_benny_hill else 'Not added'}")
    print(f"  Format: {'Vertical (1080x1920, zoomed)' if vertical_format else 'Horizontal (from video)'}")
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

    # Set dimensions based on format
    if vertical_format:
        profile_width = 1080
        profile_height = 1920
        aspect_num = '9'
        aspect_den = '16'
    else:
        profile_width = video_info['width']
        profile_height = video_info['height']
        aspect_num = '16'
        aspect_den = '9'

    profile = etree.SubElement(mlt, 'profile',
                               description='automatic',
                               width=str(profile_width),
                               height=str(profile_height),
                               progressive='1',
                               sample_aspect_num='1',
                               sample_aspect_den='1',
                               display_aspect_num=aspect_num,
                               display_aspect_den=aspect_den,
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

    # We'll create individual chains for each clip segment
    video_name = Path(video_path).stem
    chains = []  # Store chain elements for later reference

    # Create video playlist
    playlist0 = etree.SubElement(mlt, 'playlist', id='playlist0')
    etree.SubElement(playlist0, 'property', name='shotcut:video').text = '1'
    etree.SubElement(playlist0, 'property', name='shotcut:name').text = 'V1'
    etree.SubElement(playlist0, 'property', name='meta.shotcut.vui').text = '1'

    if mode == 'cut_markers':
        # Mode 1: Full video with cut markers at event boundaries
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

        # Create segments between cuts
        if cut_points:
            # Create segments between cuts
            segment_starts = [0.0] + cut_points
            segment_ends = cut_points + [video_duration]

            for i, (start, end) in enumerate(zip(segment_starts, segment_ends)):
                # Create a separate chain for each segment
                chain_id = f'chain{i}'
                start_tc = format_timecode(start, fps)
                end_tc = format_timecode(end, fps)

                chain = etree.SubElement(mlt, 'chain', id=chain_id, out=end_tc)
                etree.SubElement(chain, 'property', name='length').text = end_tc
                etree.SubElement(chain, 'property', name='eof').text = 'pause'
                etree.SubElement(chain, 'property', name='resource').text = video_path
                etree.SubElement(chain, 'property', name='mlt_service').text = 'avformat-novalidate'
                etree.SubElement(chain, 'property', name='shotcut:caption').text = video_name
                etree.SubElement(chain, 'property', name='xml').text = 'was here'

                # Disable audio if mute_audio is enabled
                if mute_audio:
                    etree.SubElement(chain, 'property', name='audio_index').text = '-1'

                # Insert chain before playlist (MLT convention)
                playlist0.addprevious(chain)

                # Add playlist entry referencing this chain
                entry = etree.SubElement(playlist0, 'entry',
                                        producer=chain_id,
                                        **{'in': start_tc, 'out': end_tc})

            print(f"\nCreated {len(segment_starts)} segments with {len(cut_points)} cut points")
        else:
            # No cuts - just add the entire video as one segment
            chain_id = 'chain0'
            start_tc = format_timecode(0.0, fps)
            end_tc = format_timecode(video_duration, fps)

            chain = etree.SubElement(mlt, 'chain', id=chain_id, out=end_tc)
            etree.SubElement(chain, 'property', name='length').text = end_tc
            etree.SubElement(chain, 'property', name='eof').text = 'pause'
            etree.SubElement(chain, 'property', name='resource').text = video_path
            etree.SubElement(chain, 'property', name='mlt_service').text = 'avformat-novalidate'
            etree.SubElement(chain, 'property', name='shotcut:caption').text = video_name
            etree.SubElement(chain, 'property', name='xml').text = 'was here'

            # Disable audio if mute_audio is enabled
            if mute_audio:
                etree.SubElement(chain, 'property', name='audio_index').text = '-1'

            # Insert chain before playlist
            playlist0.addprevious(chain)

            entry = etree.SubElement(playlist0, 'entry',
                                    producer=chain_id,
                                    **{'in': start_tc, 'out': end_tc})
            print(f"\nCreated 1 segment (no cuts)")

        # Timeline duration is the full video duration
        total_duration_tc = video_duration_tc

    else:  # mode == 'extract_clips'
        # Mode 2: Extract event clips only, concatenated back-to-back (montage)
        # Step 1: Calculate clip boundaries with buffers for all events
        clips = []

        for i, event in enumerate(events):
            # Calculate clip boundaries with buffers
            clip_start = max(0, event['start_time'] - marker_buffer)
            clip_end = min(video_duration, event['end_time'] + post_buffer)
            clip_duration = clip_end - clip_start

            if clip_duration <= 0:
                print(f"  Skipped event {i+1}: invalid clip duration")
                continue

            clips.append({
                'event_id': event.get('event_id', i + 1),
                'start': clip_start,
                'end': clip_end,
                'duration': clip_duration
            })

            print(f"  Event {i+1}: {format_timecode(clip_start, fps)} → {format_timecode(clip_end, fps)} (duration: {clip_duration:.2f}s)")

        # Step 2: Merge overlapping clips to prevent duplicate content
        print(f"\nMerging overlapping clips...")
        merged_clips = merge_overlapping_clips(clips, fps)

        if len(merged_clips) < len(clips):
            print(f"✓ Reduced {len(clips)} clips to {len(merged_clips)} clips after merging")
        else:
            print(f"✓ No overlapping clips detected ({len(clips)} clips)")

        # Step 3: Create MLT chains from merged clips
        print(f"\nCreating timeline clips...")
        total_montage_duration = 0.0

        for chain_index, clip in enumerate(merged_clips):
            clip_start = clip['start']
            clip_end = clip['end']
            clip_duration = clip['duration']

            # Create a separate chain for each clip
            chain_id = f'chain{chain_index}'
            start_tc = format_timecode(clip_start, fps)
            end_tc = format_timecode(clip_end, fps)

            chain = etree.SubElement(mlt, 'chain', id=chain_id, out=end_tc)
            etree.SubElement(chain, 'property', name='length').text = end_tc
            etree.SubElement(chain, 'property', name='eof').text = 'pause'
            etree.SubElement(chain, 'property', name='resource').text = video_path
            etree.SubElement(chain, 'property', name='mlt_service').text = 'avformat-novalidate'
            etree.SubElement(chain, 'property', name='shotcut:caption').text = video_name
            etree.SubElement(chain, 'property', name='xml').text = 'was here'

            # Disable audio if mute_audio is enabled
            if mute_audio:
                etree.SubElement(chain, 'property', name='audio_index').text = '-1'

            # Insert chain before playlist (MLT convention)
            playlist0.addprevious(chain)

            # Add playlist entry for this clip
            entry = etree.SubElement(playlist0, 'entry',
                                    producer=chain_id,
                                    **{'in': start_tc, 'out': end_tc})

            total_montage_duration += clip_duration

            # Log which events are in this clip
            event_ids = clip.get('event_ids', [clip.get('event_id', chain_index + 1)])
            if len(event_ids) > 1:
                print(f"  Clip {chain_index+1} (merged): Events {event_ids} → {format_timecode(clip_start, fps)} - {format_timecode(clip_end, fps)} ({clip_duration:.2f}s)")
            else:
                print(f"  Clip {chain_index+1}: Event {event_ids[0]} → {format_timecode(clip_start, fps)} - {format_timecode(clip_end, fps)} ({clip_duration:.2f}s)")

        print(f"\nCreated {len(merged_clips)} clips with total montage duration: {total_montage_duration:.2f}s")

        # Timeline duration is the sum of all clip durations
        total_duration_tc = format_timecode(total_montage_duration, fps)

    # Add affine filter for vertical format (zoom/scale)
    if vertical_format:
        affine_filter = etree.SubElement(playlist0, 'filter', id='filter0', out=total_duration_tc)
        etree.SubElement(affine_filter, 'property', name='background').text = 'color:#00000000'
        etree.SubElement(affine_filter, 'property', name='mlt_service').text = 'affine'
        etree.SubElement(affine_filter, 'property', name='shotcut:filter').text = 'affineSizePosition'
        etree.SubElement(affine_filter, 'property', name='transition.fix_rotate_x').text = '0'
        etree.SubElement(affine_filter, 'property', name='transition.fill').text = '1'
        etree.SubElement(affine_filter, 'property', name='transition.distort').text = '0'
        etree.SubElement(affine_filter, 'property', name='transition.rect').text = '-1122.45 -1998.5 3328.56 5917 1'
        etree.SubElement(affine_filter, 'property', name='transition.valign').text = 'middle'
        etree.SubElement(affine_filter, 'property', name='transition.halign').text = 'center'
        etree.SubElement(affine_filter, 'property', name='shotcut:animIn').text = '00:00:00.000'
        etree.SubElement(affine_filter, 'property', name='shotcut:animOut').text = '00:00:00.000'
        etree.SubElement(affine_filter, 'property', name='transition.threads').text = '0'

    # Add Benny Hill audio track if requested
    if add_benny_hill:
        # Get absolute path to Benny Hill audio file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        benny_hill_path = os.path.join(script_dir, '..', 'static', 'Benny-hill-theme.mp3')
        benny_hill_path = os.path.abspath(benny_hill_path)

        # Constants for YouTube Shorts
        max_duration = 59.0  # seconds
        fade_duration = 3.0  # seconds
        fade_start = max_duration - fade_duration  # 56 seconds

        # Create audio producer (chain)
        audio_chain = etree.SubElement(mlt, 'chain', id='chain-benny-hill', out=format_timecode(max_duration, fps))
        etree.SubElement(audio_chain, 'property', name='length').text = format_timecode(max_duration, fps)
        etree.SubElement(audio_chain, 'property', name='eof').text = 'pause'
        etree.SubElement(audio_chain, 'property', name='resource').text = benny_hill_path
        etree.SubElement(audio_chain, 'property', name='mlt_service').text = 'avformat-novalidate'
        etree.SubElement(audio_chain, 'property', name='audio_index').text = '0'
        etree.SubElement(audio_chain, 'property', name='video_index').text = '-1'  # Audio only
        etree.SubElement(audio_chain, 'property', name='shotcut:caption').text = 'Benny Hill Theme'

        # Add fade-out filter (volume with keyframes)
        fadeout_filter = etree.SubElement(audio_chain, 'filter', id='filter-fadeout', out=format_timecode(max_duration, fps))
        etree.SubElement(fadeout_filter, 'property', name='window').text = '75'
        etree.SubElement(fadeout_filter, 'property', name='max_gain').text = '20dB'
        # Keyframed level: fade from 0dB (full volume) to -60dB (silence)
        fade_start_tc = format_timecode(fade_start, fps)
        fade_end_tc = format_timecode(max_duration, fps)
        etree.SubElement(fadeout_filter, 'property', name='level').text = f'{fade_start_tc}=0;{fade_end_tc}=-60'
        etree.SubElement(fadeout_filter, 'property', name='channel_mask').text = '-1'
        etree.SubElement(fadeout_filter, 'property', name='mlt_service').text = 'volume'
        etree.SubElement(fadeout_filter, 'property', name='shotcut:filter').text = 'fadeOutVolume'
        etree.SubElement(fadeout_filter, 'property', name='shotcut:animOut').text = format_timecode(fade_duration, fps)

        # Create audio playlist
        playlist_audio = etree.SubElement(mlt, 'playlist', id='playlist-audio')
        etree.SubElement(playlist_audio, 'property', name='shotcut:audio').text = '1'
        etree.SubElement(playlist_audio, 'property', name='shotcut:name').text = 'A1'
        etree.SubElement(playlist_audio, 'entry',
                        producer='chain-benny-hill',
                        **{'in': '00:00:00.000', 'out': format_timecode(max_duration, fps)})

    # Create tractor combining background and video tracks
    tractor = etree.SubElement(mlt, 'tractor',
                               id='tractor0',
                               title='Shotcut version 25.08.16',
                               **{'in': '00:00:00.000', 'out': total_duration_tc})
    etree.SubElement(tractor, 'property', name='shotcut').text = '1'
    etree.SubElement(tractor, 'property', name='shotcut:projectAudioChannels').text = '2'
    etree.SubElement(tractor, 'property', name='shotcut:projectFolder').text = '1'
    etree.SubElement(tractor, 'property', name='shotcut:skipConvert').text = '0'

    # Add tracks: background + video playlist + optional audio
    etree.SubElement(tractor, 'track', producer='background')
    etree.SubElement(tractor, 'track', producer='playlist0')
    if add_benny_hill:
        etree.SubElement(tractor, 'track', producer='playlist-audio', hide='video')

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

    # Add audio mix transition for Benny Hill track if present
    if add_benny_hill:
        transition_audio = etree.SubElement(tractor, 'transition', id='transition-audio-mix')
        etree.SubElement(transition_audio, 'property', name='a_track').text = '0'
        etree.SubElement(transition_audio, 'property', name='b_track').text = '2'
        etree.SubElement(transition_audio, 'property', name='mlt_service').text = 'mix'
        etree.SubElement(transition_audio, 'property', name='always_active').text = '1'
        etree.SubElement(transition_audio, 'property', name='sum').text = '1'

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

    if mode == 'cut_markers':
        print(f"Video split into {len(segment_starts)} segments at {len(cut_points)} cut points")
        print(f"\nTo use:")
        print(f"  1. Open Shotcut: shotcut {output_path}")
        print(f"  2. Timeline shows the full video with cuts at predicted moments")
        print(f"  3. Manually delete unwanted segments or File → Export to render")
    else:
        print(f"Montage contains {len(merged_clips)} clips")
        print(f"\nTo use:")
        print(f"  1. Open Shotcut: shotcut {output_path}")
        print(f"  2. Timeline shows extracted event clips concatenated back-to-back")
        print(f"  3. File → Export → choose format and render montage")


def generate_from_events_file(events_file: str, output_path: str = None,
                             marker_buffer: float = None, post_buffer: float = None,
                             mode: str = None, mute_audio: bool = None, add_benny_hill: bool = None,
                             vertical_format: bool = None, config_path: str = 'config.yaml'):
    """
    Generate MLT project from events JSON file.

    Args:
        events_file: Path to events JSON file
        output_path: Output MLT path (optional)
        marker_buffer: Marker buffer in seconds (optional, defaults to config value)
        post_buffer: Post-event buffer in seconds (optional, defaults to config value)
        mode: Generation mode 'cut_markers' or 'extract_clips' (optional, defaults to 'cut_markers')
        mute_audio: Mute source video's audio track (optional, defaults to config value)
        add_benny_hill: Add Benny Hill theme song (optional, defaults to config value)
        vertical_format: Use vertical 1080x1920 format with zoom (optional, defaults to config value)
        config_path: Path to config file (default: config.yaml)
    """
    # Load config
    config = load_config(config_path)

    # Use provided values or fall back to config
    if marker_buffer is None:
        marker_buffer = config.get('mlt', {}).get('marker_buffer', 2.0)
    if post_buffer is None:
        post_buffer = config.get('mlt', {}).get('post_buffer', 1.0)
    if mode is None:
        mode = 'cut_markers'
    if mute_audio is None:
        mute_audio = config.get('mlt', {}).get('mute_audio', False)
    if add_benny_hill is None:
        add_benny_hill = config.get('mlt', {}).get('add_benny_hill', False)
    if vertical_format is None:
        vertical_format = config.get('mlt', {}).get('vertical_format', False)

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
        output_path = get_experiment_path(video_name)

    # Generate MLT
    create_mlt_project(video_path, events, fps, output_path, marker_buffer, post_buffer, mode, mute_audio, add_benny_hill, vertical_format)


def main():
    parser = argparse.ArgumentParser(description='Generate Shotcut MLT project')
    parser.add_argument('--events', required=True, help='Path to events JSON file')
    parser.add_argument('--output', help='Output MLT path (default: outputs/projects/<video>_project.mlt)')
    parser.add_argument('--marker-buffer', type=float, help='Pre-event buffer in seconds (default: from config)')
    parser.add_argument('--post-buffer', type=float, help='Post-event buffer in seconds (default: from config)')
    parser.add_argument('--mode', choices=['cut_markers', 'extract_clips'], default='cut_markers',
                       help='Generation mode: cut_markers (full video with cuts) or extract_clips (montage)')

    # Mutually exclusive audio flags
    audio_group = parser.add_mutually_exclusive_group()
    audio_group.add_argument('--mute-audio', action='store_true', dest='mute_audio_flag', help='Mute source video audio')
    audio_group.add_argument('--no-mute-audio', action='store_true', dest='no_mute_audio_flag', help='Enable source video audio')

    # Benny Hill theme song flag
    benny_group = parser.add_mutually_exclusive_group()
    benny_group.add_argument('--add-benny-hill', action='store_true', dest='add_benny_hill_flag', help='Add Benny Hill theme song')
    benny_group.add_argument('--no-benny-hill', action='store_true', dest='no_benny_hill_flag', help='Do not add Benny Hill theme song')

    # Vertical format flag
    parser.add_argument('--vertical-format', action='store_true', dest='vertical_format_flag', help='Use vertical 1080x1920 format with zoom')

    parser.add_argument('--config', default='config.yaml', help='Config file path')

    args = parser.parse_args()

    # Determine mute_audio: explicit flag overrides, otherwise None (uses config default)
    if args.mute_audio_flag:
        mute_audio = True
    elif args.no_mute_audio_flag:
        mute_audio = False
    else:
        mute_audio = None

    # Determine add_benny_hill: explicit flag overrides, otherwise None (uses config default)
    if args.add_benny_hill_flag:
        add_benny_hill = True
    elif args.no_benny_hill_flag:
        add_benny_hill = False
    else:
        add_benny_hill = None

    # Determine vertical_format: explicit flag overrides, otherwise None (uses config default)
    vertical_format = True if args.vertical_format_flag else None

    generate_from_events_file(args.events, args.output, args.marker_buffer,
                             args.post_buffer, args.mode, mute_audio, add_benny_hill, vertical_format, args.config)


if __name__ == '__main__':
    main()
