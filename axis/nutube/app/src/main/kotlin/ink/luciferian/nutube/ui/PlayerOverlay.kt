package ink.luciferian.nutube.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.media3.common.MediaItem
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.ui.PlayerView
import ink.luciferian.nutube.data.FeedItem
import ink.luciferian.nutube.source.YouTubeSource

/**
 * In-app playback: ExoPlayer hardware-decodes the stream onto its own SurfaceView,
 * which Android can hand straight to the display controller. This is the capability
 * the Godot build could not have - VideoStreamPlayer only speaks Ogg Theora.
 */
@Composable
fun PlayerOverlay(item: FeedItem, onClose: () -> Unit) {
	val context = LocalContext.current
	var streamUrl by remember(item.id) { mutableStateOf<String?>(null) }
	var failure by remember(item.id) { mutableStateOf<String?>(null) }

	LaunchedEffect(item.id) {
		YouTubeSource.playbackUrl(item.id)
			.onSuccess { streamUrl = it }
			.onFailure { failure = it.message ?: "could not resolve a stream" }
	}

	val player = remember {
		ExoPlayer.Builder(context).build().apply { playWhenReady = true }
	}

	LaunchedEffect(streamUrl) {
		streamUrl?.let {
			player.setMediaItem(MediaItem.fromUri(it))
			player.prepare()
		}
	}

	DisposableEffect(Unit) {
		onDispose { player.release() }
	}

	Box(
		Modifier.fillMaxSize().background(Color.Black.copy(alpha = 0.94f)),
		contentAlignment = Alignment.Center,
	) {
		Column(Modifier.fillMaxWidth()) {
			when {
				failure != null -> Text(
					failure!!,
					Modifier.padding(24.dp),
					style = MaterialTheme.typography.bodyMedium,
				)

				streamUrl == null -> Box(
					Modifier.fillMaxWidth().aspectRatio(16f / 9f),
					contentAlignment = Alignment.Center,
				) { CircularProgressIndicator() }

				else -> AndroidView(
					factory = { PlayerView(it).apply { this.player = player } },
					modifier = Modifier.fillMaxWidth().aspectRatio(16f / 9f),
				)
			}
			Text(
				item.title,
				Modifier.padding(horizontal = 16.dp, vertical = 8.dp),
				style = MaterialTheme.typography.titleMedium,
			)
			if (item.author.isNotEmpty()) {
				Text(
					item.author,
					Modifier.padding(horizontal = 16.dp),
					style = MaterialTheme.typography.bodySmall,
					color = Color.Gray,
				)
			}
			TextButton(onClick = onClose, modifier = Modifier.padding(8.dp)) { Text("Back") }
		}
	}
}
