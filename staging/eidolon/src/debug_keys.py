#!/usr/bin/env python3
"""
Debug script to discover key codes on your system.
Press keys to see their codes. Press ESC to exit.
"""

import cv2
import numpy as np

def main():
    print("Key Code Debugger")
    print("=" * 50)
    print("Press keys to see their codes")
    print("Press ESC to exit")
    print("=" * 50)
    print()

    # Create a simple window
    img = np.zeros((300, 600, 3), dtype=np.uint8)
    cv2.putText(img, "Press keys to see codes", (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    window_name = "Key Debugger"
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, img)

    while True:
        # Try both waitKey and waitKeyEx
        key_ex = cv2.waitKeyEx(0)
        key_char = key_ex & 0xFF

        # ESC to exit
        if key_char == 27:
            print("\nExiting...")
            break

        print(f"Key pressed:")
        print(f"  waitKeyEx(): {key_ex} (0x{key_ex:X})")
        print(f"  8-bit mask:  {key_char} (0x{key_char:X})")
        print(f"  chr({key_char}): '{chr(key_char) if 32 <= key_char < 127 else '?'}'")
        print()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
