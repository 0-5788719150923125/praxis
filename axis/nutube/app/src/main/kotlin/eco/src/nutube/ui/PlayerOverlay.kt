package eco.src.nutube.ui

import androidx.annotation.OptIn
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.media3.common.MediaItem
import androidx.media3.common.util.UnstableApi
import androidx.media3.datasource.DefaultHttpDataSource
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.exoplayer.source.MergingMediaSource
import androidx.media3.exoplayer.source.ProgressiveMediaSource
import androidx.media3.ui.PlayerView
import eco.src.nutube.core.FeedItem
import eco.src.nutube.core.PlaybackStreams
import eco.src.nutube.core.SourceRegistry

/**
 * In-app playback, hardware-decoded onto ExoPlayer's own SurfaceView.
 *
 * The HD path is [PlaybackStreams.Split]: YouTube caps its muxed streams at 720p,
 * so anything better arrives as a video-only track plus a separate audio track.
 * [MergingMediaSource] plays them as one, keeping them in sync.
 */
@OptIn(UnstableApi::class)
@Composable
fun PlayerOverlay(item: FeedItem, onClose: () -> Unit) {
	val context = LocalContext.current
	var streams by remember(item.key) { mutableStateOf<PlaybackStreams?>(null) }
	var failure by remember(item.key) { mutableStateOf<String?>(null) }

	LaunchedEffect(item.key) {
		val source = SourceRegistry.forItem(item)
		if (source == null) {
			failure = "no source registered for '${item.source}'"
			return@LaunchedEffect
		}
		source.streams(item.id)
			.onSuccess { streams = it }
			.onFailure { failure = it.message ?: "could not resolve a stream" }
	}

	val player = remember { ExoPlayer.Builder(context).build().apply { playWhenReady = true } }

	LaunchedEffect(streams) {
		val current = streams ?: return@LaunchedEffect
		// googlevideo rejects requests without a browser-shaped User-Agent.
		val http = DefaultHttpDataSource.Factory()
			.setUserAgent("Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0")
			.setAllowCrossProtocolRedirects(true)

		when (current) {
			is PlaybackStreams.Single ->
				player.setMediaItem(MediaItem.fromUri(current.url))

			is PlaybackStreams.Split -> {
				val factory = ProgressiveMediaSource.Factory(http)
				player.setMediaSource(
					MergingMediaSource(
						factory.createMediaSource(MediaItem.fromUri(current.videoUrl)),
						factory.createMediaSource(MediaItem.fromUri(current.audioUrl)),
					)
				)
			}
		}
		player.prepare()
	}

	DisposableEffect(Unit) { onDispose { player.release() } }

	Box(
		Modifier.fillMaxSize().background(Ink.copy(alpha = 0.97f)),
		contentAlignment = Alignment.Center,
	) {
		Column(Modifier.fillMaxWidth()) {
			when {
				failure != null -> Text(
					failure!!,
					Modifier.padding(24.dp),
					style = MaterialTheme.typography.bodyMedium,
					color = Muted,
				)

				streams == null -> Box(
					Modifier.fillMaxWidth().aspectRatio(16f / 9f),
					contentAlignment = Alignment.Center,
				) { CircularProgressIndicator(color = Accent) }

				else -> AndroidView(
					factory = { PlayerView(it).apply { this.player = player } },
					modifier = Modifier.fillMaxWidth().aspectRatio(16f / 9f),
				)
			}

			Text(
				item.title,
				Modifier.padding(start = 16.dp, end = 16.dp, top = 14.dp),
				style = MaterialTheme.typography.titleMedium,
				color = Bright,
			)
			Row(
				Modifier.padding(horizontal = 16.dp, vertical = 4.dp),
			) {
				if (item.author.isNotEmpty()) {
					Text(item.author, style = MaterialTheme.typography.bodySmall, color = Muted)
				}
				val quality = when (val s = streams) {
					is PlaybackStreams.Split -> s.label
					is PlaybackStreams.Single -> s.label
					null -> ""
				}
				if (quality.isNotEmpty()) {
					Text(
						"  ·  $quality",
						style = MaterialTheme.typography.bodySmall,
						color = Accent.copy(alpha = 0.8f),
					)
				}
			}
			TextButton(onClick = onClose, modifier = Modifier.padding(8.dp)) {
				Text("Back", color = Bright)
			}
		}
	}
}
