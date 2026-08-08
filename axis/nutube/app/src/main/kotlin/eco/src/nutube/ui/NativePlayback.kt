package eco.src.nutube.ui

import android.content.Context
import androidx.annotation.OptIn
import androidx.media3.common.MediaItem
import androidx.media3.common.PlaybackException
import androidx.media3.common.Player
import androidx.media3.common.VideoSize
import androidx.media3.common.util.UnstableApi
import androidx.media3.datasource.DefaultHttpDataSource
import androidx.media3.exoplayer.DefaultLoadControl
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.exoplayer.source.MergingMediaSource
import androidx.media3.exoplayer.source.ProgressiveMediaSource
import eco.src.nutube.core.PlaybackStreams
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Owns the native player, outliving the composables that show it.
 *
 * The player deliberately does not live inside the player composable. Entering
 * picture-in-picture rearranges the UI tree, and a player owned by that tree gets
 * torn down and rebuilt with it - which is why PiP used to restart every video
 * from zero. Held here, the same instance survives the transition and keeps its
 * position.
 *
 * [prepare] is keyed and idempotent for the same video, so any number of
 * recompositions cannot restart what is already playing.
 */
@OptIn(UnstableApi::class)
class NativePlayback(context: Context) {

	private var preparedKey: String? = null

	private val _error = MutableStateFlow<String?>(null)
	val error: StateFlow<String?> = _error.asStateFlow()

	/** Width / height of the current video, so the surface can be sized to fit it. */
	private val _aspect = MutableStateFlow(16f / 9f)
	val aspect: StateFlow<Float> = _aspect.asStateFlow()

	val player: ExoPlayer = ExoPlayer.Builder(context)
		// The defaults are tuned for local media. Streaming googlevideo over a phone
		// connection wants a deeper buffer before it starts, and much more headroom
		// after, or the first stall never recovers.
		.setLoadControl(
			DefaultLoadControl.Builder()
				.setBufferDurationsMs(30_000, 120_000, 2_500, 5_000)
				.build()
		)
		.build()
		.apply {
			playWhenReady = true
			addListener(object : Player.Listener {
				override fun onPlayerError(e: PlaybackException) {
					_error.value = e.errorCodeName + (e.message?.let { ": $it" } ?: "")
				}

				override fun onVideoSizeChanged(size: VideoSize) {
					if (size.width > 0 && size.height > 0) {
						_aspect.value =
							size.width * size.pixelWidthHeightRatio / size.height.toFloat()
					}
				}
			})
		}

	fun prepare(key: String, streams: PlaybackStreams) {
		if (preparedKey == key) return
		preparedKey = key
		_error.value = null

		val http = DefaultHttpDataSource.Factory()
			.setUserAgent("Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0")
			.setAllowCrossProtocolRedirects(true)
			// ExoPlayer opens at position 0 with no length, so it sends no Range
			// header, and googlevideo throttles an open-ended GET to a trickle after
			// the first few seconds. A default Range makes the opening request a
			// ranged one; the player only overrides this when it computes its own.
			.setDefaultRequestProperties(mapOf("Range" to "bytes=0-"))
		val factory = ProgressiveMediaSource.Factory(http)

		player.setMediaSource(
			when (streams) {
				is PlaybackStreams.Single ->
					factory.createMediaSource(MediaItem.fromUri(streams.url))

				// Above 720p YouTube publishes video and audio separately; merging
				// them is what makes HD possible.
				is PlaybackStreams.Split -> MergingMediaSource(
					factory.createMediaSource(MediaItem.fromUri(streams.videoUrl)),
					factory.createMediaSource(MediaItem.fromUri(streams.audioUrl)),
				)
			}
		)
		player.playWhenReady = true
		player.prepare()
	}

	/** Closing the player: drop the media so its codecs go back to the system. */
	fun stop() {
		preparedKey = null
		_error.value = null
		_aspect.value = 16f / 9f
		player.stop()
		player.clearMediaItems()
		player.clearVideoSurface()
	}

	fun release() {
		preparedKey = null
		player.release()
	}
}
