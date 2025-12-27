#!/usr/bin/env python3
"""
Download videos from URLs using yt-dlp.

Downloads at 1080p maximum with h264 video codec and AAC audio.

Usage:
    python src/download_video.py --url "https://youtube.com/watch?v=..." --output videos/
"""

import os
import sys
import argparse
from pathlib import Path

try:
    from yt_dlp import YoutubeDL
except ImportError:
    print("ERROR: yt-dlp is not installed. Please run: pip install yt-dlp")
    sys.exit(1)


def progress_hook(d):
    """
    Progress hook for yt-dlp to print download status.

    This output is captured by the GUI's run_command() for real-time display.
    """
    if d['status'] == 'downloading':
        percent = d.get('_percent_str', 'N/A').strip()
        speed = d.get('_speed_str', 'N/A').strip()
        eta = d.get('_eta_str', 'N/A').strip()
        print(f"Downloading: {percent} at {speed}, ETA: {eta}", flush=True)

    elif d['status'] == 'finished':
        filename = os.path.basename(d.get('filename', 'unknown'))
        print(f"Download complete: {filename}", flush=True)
        print("Processing video...", flush=True)


def download_video(url: str, output_dir: str, format_spec: str = 'bestvideo[height<=1080][vcodec^=avc1]+bestaudio[acodec^=mp4a]/best[height<=1080]'):
    """
    Download video from URL using yt-dlp.

    Args:
        url: Video URL (YouTube, Vimeo, etc.)
        output_dir: Output directory for downloaded video
        format_spec: Format specifier (default: 1080p max, h264 video + AAC audio)

    Returns:
        Path to downloaded file
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting download from: {url}")
    print(f"Output directory: {output_dir}")
    print(f"Format: {format_spec}")
    print()

    # Configure yt-dlp options
    ydl_opts = {
        'format': format_spec,
        'outtmpl': os.path.join(output_dir, '%(title)s - %(uploader)s (%(height)sp, %(vcodec)s).%(ext)s'),
        'progress_hooks': [progress_hook],
        'quiet': False,
        'no_warnings': False,
        'ignoreerrors': False,
        'extract_flat': False,
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            # Extract info first (without download) to show video details
            print("Fetching video information...")
            info = ydl.extract_info(url, download=False)

            # Show video details
            title = info.get('title', 'Unknown')
            duration = info.get('duration', 0)
            uploader = info.get('uploader', 'Unknown')
            height = info.get('height', 0)

            print(f"\nVideo: {title}")
            print(f"Uploader: {uploader}")
            print(f"Duration: {duration // 60}:{duration % 60:02d}")
            print(f"Resolution: {height}p")
            print()

            # Now download
            print("Starting download...")
            info = ydl.extract_info(url, download=True)

            # Get the actual downloaded filename
            filename = ydl.prepare_filename(info)

            # Normalize codec name: avc1.* → h264, hev1.*/hvc1.* → h265, etc.
            if os.path.exists(filename):
                vcodec = info.get('vcodec', 'unknown')

                # Map codec strings to friendly names
                codec_normalized = 'unknown'
                if vcodec and isinstance(vcodec, str):
                    if vcodec.startswith('avc1'):
                        codec_normalized = 'h264'
                    elif vcodec.startswith('hev1') or vcodec.startswith('hvc1'):
                        codec_normalized = 'h265'
                    elif vcodec.startswith('vp9'):
                        codec_normalized = 'vp9'
                    elif vcodec.startswith('av01'):
                        codec_normalized = 'av1'
                    else:
                        codec_normalized = vcodec.split('.')[0]  # Take first part

                # Build proper filename: <Title> - <Uploader> (<resolution>, <codec>).ext
                title_clean = info.get('title', 'video')
                uploader_clean = info.get('uploader', 'Unknown')
                ext = os.path.splitext(filename)[1]

                new_basename = f"{title_clean} - {uploader_clean} ({height}p, {codec_normalized}){ext}"
                new_filename = os.path.join(output_dir, new_basename)

                # Rename if different
                if filename != new_filename:
                    print(f"\nNormalizing filename to match convention...")
                    print(f"  Old: {os.path.basename(filename)}")
                    print(f"  New: {os.path.basename(new_basename)}")

                    # Handle existing file
                    if os.path.exists(new_filename):
                        print(f"  Target file already exists, removing temporary download...")
                        os.remove(filename)
                        filename = new_filename
                    else:
                        os.rename(filename, new_filename)
                        filename = new_filename
                        print(f"  ✓ Renamed successfully")

            print()
            print("=" * 80)
            print("SUCCESS!")
            print("=" * 80)
            print(f"Downloaded: {os.path.basename(filename)}")
            print(f"Location: {filename}")
            print()

            return filename

    except Exception as e:
        print()
        print("=" * 80)
        print("ERROR!")
        print("=" * 80)
        print(f"Failed to download video: {e}")
        print()
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Download video from URL using yt-dlp'
    )
    parser.add_argument(
        '--url',
        required=True,
        help='Video URL (YouTube, Vimeo, etc.)'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output directory for downloaded video'
    )
    parser.add_argument(
        '--format',
        default='bestvideo[height<=1080][vcodec^=avc1]+bestaudio[acodec^=mp4a]/best[height<=1080]',
        help='Format specifier (default: 1080p max, h264 video + AAC audio)'
    )

    args = parser.parse_args()

    try:
        download_video(args.url, args.output, args.format)
        sys.exit(0)
    except Exception:
        sys.exit(1)


if __name__ == '__main__':
    main()
