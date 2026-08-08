package eco.src.nutube.core

/**
 * What a source hands back when asked how to play something.
 *
 * The split case exists because platforms that adapt bitrate publish video and
 * audio as separate tracks, and the muxed stream they also offer is a low-quality
 * fallback - on YouTube it is capped at 720p. Playing in HD means taking both
 * tracks and letting the player merge them.
 */
sealed interface PlaybackStreams {

	/** One URL that already carries both tracks: a progressive file, or an HLS/DASH manifest. */
	data class Single(val url: String, val label: String = "") : PlaybackStreams

	/** Separate video-only and audio-only tracks for the player to merge. */
	data class Split(
		val videoUrl: String,
		val audioUrl: String,
		val height: Int,
		val label: String = "",
	) : PlaybackStreams
}
