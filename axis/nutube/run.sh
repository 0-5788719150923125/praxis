#!/usr/bin/env bash
#
# Build, install and launch nuTube on an emulator - the rough equivalent of
# `godot --path axis/nutube/godot`.
#
#   ./run.sh                    boot the AVD if needed, build, install, launch, tail logs
#   ./run.sh --quiet            same, but do not tail logcat
#   ./run.sh --wipe             cold-boot the AVD with fresh data first
#   ./run.sh --headless         run the emulator with no window (useful over ssh)
#   ./run.sh --logs             just tail logs against whatever is already running
#   ./run.sh --url <youtube>    launch via a VIEW intent, to exercise the deep link
#   ./run.sh --apk              just build and drop nutube.apk here, no device needed
#   ./run.sh --stop             shut the emulator down
#
# Uses whatever device is already connected, so plugging in a real phone and
# running this unchanged installs there instead.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# The exported ANDROID_HOME on this machine points at a path that does not
# exist, so resolve the SDK ourselves rather than trusting the environment.
SDK="${NUTUBE_SDK:-$HOME/Android/Sdk}"
[ -d "$SDK" ] || { echo "no Android SDK at $SDK - set NUTUBE_SDK" >&2; exit 1; }

# AGP rejects the system default JDK; pin the one it wants.
export JAVA_HOME="${JAVA_HOME_17:-/usr/lib/jvm/java-17-openjdk}"
export ANDROID_HOME="$SDK" ANDROID_SDK_ROOT="$SDK"

ADB="$SDK/platform-tools/adb"
EMULATOR="$SDK/emulator/emulator"
AVD="${NUTUBE_AVD:-Medium_Phone_API_35}"
PKG=eco.src.nutube
ACTIVITY="$PKG/.MainActivity"

TAIL_LOGS=1
WIPE=0
APK_ONLY=0
WINDOW="-gpu host"
URL=""

while [ $# -gt 0 ]; do
	case "$1" in
		--quiet)    TAIL_LOGS=0 ;;
		--wipe)     WIPE=1 ;;
		--headless) WINDOW="-no-window -gpu swiftshader_indirect" ;;
		--logs)     exec "$ADB" logcat -v color -s "$PKG:V" AndroidRuntime:E ExoPlayer:I ;;
		--url)      shift; URL="${1:-}" ;;
		--apk)      APK_ONLY=1 ;;
		--stop)     "$ADB" emu kill 2>/dev/null || true; echo "emulator stopped"; exit 0 ;;
		-h|--help)  sed -n '3,20p' "$0" | sed 's/^# \?//'; exit 0 ;;
		*)          echo "unknown flag: $1" >&2; exit 1 ;;
	esac
	shift
done

# Syncthing excludes build/ directories, so park a copy where it will sync.
publish_apk() {
	local built=app/build/outputs/apk/debug/app-debug.apk
	[ -f "$built" ] || { echo "no APK at $built" >&2; return 1; }
	cp "$built" nutube.apk
	echo ":: nutube.apk ($(du -h nutube.apk | cut -f1)) - syncs to the phone from here"
}

if [ "$APK_ONLY" = 1 ]; then
	echo ":: building"
	./gradlew :app:assembleDebug --console=plain -q
	publish_apk
	exit 0
fi

online() { "$ADB" devices | awk 'NR>1 && $2=="device" {print $1; exit}'; }

# --- make sure something is listening -------------------------------------

if [ -z "$(online)" ]; then
	"$EMULATOR" -list-avds | grep -qx "$AVD" || {
		echo "AVD '$AVD' not found. Available:" >&2
		"$EMULATOR" -list-avds >&2
		echo "Create one with: $SDK/cmdline-tools/latest/bin/avdmanager create avd -n NAME -k 'system-images;android-35;google_apis_playstore;x86_64'" >&2
		exit 1
	}

	# The emulator wants ~3G of real RAM. Booting it onto an already-full box
	# does not fail cleanly - it drags the whole desktop into swap thrash, so
	# refuse up front instead.
	AVAIL_MB=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
	SWAP_FREE_MB=$(awk '/SwapFree/ {print int($2/1024)}' /proc/meminfo)
	if [ "$AVAIL_MB" -lt "${NUTUBE_MIN_MB:-4500}" ]; then
		echo "only ${AVAIL_MB}MB available (${SWAP_FREE_MB}MB swap free) - the emulator needs ~3G." >&2
		echo "free some up, or plug in a phone and rerun. Common culprits:" >&2
		echo "  du -sh /tmp/* | sort -rh | head      # /tmp is tmpfs here, it costs RAM" >&2
		echo "  docker stats --no-stream" >&2
		echo "  ./gradlew --stop" >&2
		echo "override with NUTUBE_MIN_MB=0 if you know better." >&2
		exit 1
	fi

	echo ":: booting $AVD (${AVAIL_MB}MB available)"
	# Cap the guest explicitly rather than inheriting whatever the AVD was
	# created with, so one bad config cannot take the host down.
	BOOT_FLAGS=(-avd "$AVD" $WINDOW -memory "${NUTUBE_RAM:-2048}" -cores "${NUTUBE_CORES:-2}"
		-netdelay none -netspeed full)
	[ "$WIPE" = 1 ] && BOOT_FLAGS+=(-wipe-data -no-snapshot-load)
	# Detach so the emulator outlives this script; its log goes to a file.
	nohup "$EMULATOR" "${BOOT_FLAGS[@]}" >/tmp/nutube-emulator.log 2>&1 &

	echo ":: waiting for device"
	"$ADB" wait-for-device
	# wait-for-device returns as soon as adb connects, which is long before
	# Android has finished starting.
	for _ in $(seq 1 180); do
		[ "$("$ADB" shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')" = "1" ] && break
		sleep 2
	done
	[ "$("$ADB" shell getprop sys.boot_completed | tr -d '\r')" = "1" ] || {
		echo "emulator did not finish booting - see /tmp/nutube-emulator.log" >&2
		exit 1
	}
fi

DEVICE="$(online)"
echo ":: device $DEVICE"

# --- build, install, launch -----------------------------------------------

echo ":: building"
./gradlew :app:installDebug --console=plain -q

publish_apk

echo ":: launching"
if [ -n "$URL" ]; then
	"$ADB" shell am start -a android.intent.action.VIEW -d "$URL" "$ACTIVITY" >/dev/null
else
	"$ADB" shell am start -n "$ACTIVITY" >/dev/null
fi

if [ "$TAIL_LOGS" = 1 ]; then
	echo ":: logs (ctrl-c to detach - the app keeps running)"
	"$ADB" logcat -c
	exec "$ADB" logcat -v color -s "$PKG:V" AndroidRuntime:E ExoPlayer:I NewPipe:V
fi
