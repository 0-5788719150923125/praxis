package eco.src.nutube.ui

import androidx.activity.compose.BackHandler
import androidx.activity.compose.LocalActivity
import androidx.annotation.OptIn
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.delay
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import android.content.res.Configuration
import android.view.LayoutInflater
import android.view.View
import androidx.compose.ui.viewinterop.AndroidView
import eco.src.nutube.R
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.media3.common.util.UnstableApi
import androidx.media3.ui.PlayerView
import eco.src.nutube.core.FeedItem
import eco.src.nutube.core.PlaybackMode
import eco.src.nutube.core.PlaybackStreams
import eco.src.nutube.core.SourceRegistry

/**
 * Plays [item] using whichever mode this platform is set to.
 *
 * There is no back button. Closing is the system's job - the gesture or key the
 * user already knows - so [BackHandler] takes it and the overlay stays free of
 * chrome that competes with it.
 */
@Composable
fun PlayerOverlay(
	item: FeedItem,
	mode: PlaybackMode,
	inPip: Boolean,
	playback: NativePlayback,
	onClose: () -> Unit,
) {
	BackHandler(enabled = true) { onClose() }

	val embedUrl = remember(item.key, mode) {
		if (mode == PlaybackMode.EMBED) SourceRegistry.forItem(item)?.embedUrl(item.id) else null
	}

	if (embedUrl != null) {
		if (inPip) {
			Box(Modifier.fillMaxSize().background(Ink)) {
				EmbedPlayer(url = embedUrl, modifier = Modifier.fillMaxSize())
			}
			return
		}
		Box(
			Modifier.fillMaxSize().background(Ink).safeDrawingPadding(),
			contentAlignment = Alignment.Center,
		) {
			Column(Modifier.fillMaxWidth()) {
				EmbedPlayer(url = embedUrl, modifier = Modifier.fillMaxWidth().aspectRatio(16f / 9f))
				Caption(item, quality = null)
			}
		}
		return
	}

	NativePlayer(item, inPip, playback)
}

@OptIn(UnstableApi::class)
@Composable
private fun NativePlayer(
	item: FeedItem,
	inPip: Boolean,
	playback: NativePlayback,
) {
	val lifecycle = LocalLifecycleOwner.current.lifecycle
	val error by playback.error.collectAsStateWithLifecycle()
	val aspect by playback.aspect.collectAsStateWithLifecycle()

	var streams by remember(item.key) { mutableStateOf<PlaybackStreams?>(null) }
	var resolveFailure by remember(item.key) { mutableStateOf<String?>(null) }

	LaunchedEffect(item.key) {
		val source = SourceRegistry.forItem(item)
		if (source == null) {
			resolveFailure = "no source registered for '${item.source}'"
			return@LaunchedEffect
		}
		source.streams(item.id)
			.onSuccess { streams = it }
			.onFailure { resolveFailure = it.message ?: "could not resolve a stream" }
	}

	// Keyed on the item, and idempotent, so entering or leaving picture-in-picture
	// cannot restart what is already playing.
	LaunchedEffect(item.key, streams) {
		streams?.let { playback.prepare(item.key, it) }
	}

	// Backgrounding without entering picture-in-picture should stop the audio
	// rather than leave it playing from nowhere.
	DisposableEffect(lifecycle, inPip) {
		val observer = LifecycleEventObserver { _, event ->
			if (event == Lifecycle.Event.ON_STOP && !inPip) playback.player.pause()
		}
		lifecycle.addObserver(observer)
		onDispose { lifecycle.removeObserver(observer) }
	}

	val failure = resolveFailure ?: error

	// Fullscreen is a real state, not "the player is open". Outside it the bars
	// stay put, so the feed and settings are never affected; inside it they follow
	// the player's own controls, and a tap brings both back together.
	//
	// The timer is not redundant with the visibility listener. PlayerView only
	// auto-shows its controls when playback is idle, paused or ended, so during
	// normal playback there may be no visible-to-gone transition to hook and the
	// listener never fires on its own.
	// Rotating to landscape is how most people ask for fullscreen, so it counts as
	// entering it. The button is the portrait way in; either one gets the same
	// state, and the bars follow that state rather than the player merely existing.
	val landscape =
		LocalConfiguration.current.orientation == Configuration.ORIENTATION_LANDSCAPE
	var buttonFullscreen by rememberSaveable(item.key) { mutableStateOf(false) }
	val fullscreen = buttonFullscreen || landscape
	var controlsUp by remember { mutableStateOf(true) }

	val activity = LocalActivity.current
	val localView = LocalView.current
	val bars: WindowInsetsControllerCompat? = remember(activity) {
		activity?.window?.let { WindowCompat.getInsetsController(it, localView) }?.apply {
			systemBarsBehavior =
				WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
		}
	}

	LaunchedEffect(fullscreen, controlsUp, inPip) {
		val systemBars = WindowInsetsCompat.Type.systemBars()
		if (!fullscreen || inPip) {
			bars?.show(systemBars)
			return@LaunchedEffect
		}
		if (controlsUp) {
			bars?.show(systemBars)
			delay(CONTROLS_TIMEOUT_MS)
			controlsUp = false
		} else {
			bars?.hide(systemBars)
		}
	}

	// A safety net: however the player leaves, the bars come back.
	DisposableEffect(bars) {
		onDispose { bars?.show(WindowInsetsCompat.Type.systemBars()) }
	}

	// Back exits fullscreen before it closes the player, which is what the gesture
	// means in every other video app. It cannot undo a rotation, so it only
	// reverses the button.
	BackHandler(enabled = buttonFullscreen) { buttonFullscreen = false }

	val surface: @Composable (Modifier) -> Unit = { mod ->
		NativeSurface(
			playback = playback,
			useController = !inPip,
			onControlsVisible = { controlsUp = it },
			onFullscreenClick = { buttonFullscreen = !buttonFullscreen },
			modifier = mod,
		)
	}

	if (inPip) {
		Box(Modifier.fillMaxSize().background(Ink)) { surface(Modifier.fillMaxSize()) }
		return
	}

	// Fullscreen: nothing but the video, edge to edge, letterboxed to its shape.
	if (fullscreen) {
		Box(Modifier.fillMaxSize().background(Ink), contentAlignment = Alignment.Center) {
			surface(Modifier.fillMaxSize())
		}
		return
	}

	Box(
		Modifier.fillMaxSize().background(Ink.copy(alpha = 0.97f)).safeDrawingPadding(),
		contentAlignment = Alignment.Center,
	) {
		Column(Modifier.fillMaxWidth()) {
			when {
				failure != null -> Text(
					failure,
					Modifier.padding(24.dp),
					style = MaterialTheme.typography.bodyMedium,
					color = Muted,
				)

				streams == null -> Box(
					Modifier.fillMaxWidth().aspectRatio(16f / 9f),
					contentAlignment = Alignment.Center,
				) { CircularProgressIndicator(color = Accent) }

				// Size to the video's own shape rather than forcing 16:9, so a
				// portrait or 4:3 clip is neither cropped nor stranded in bars.
				// Clamped so the box is never taller than it is wide: an unclamped
				// 4:3 or portrait clip overflows the screen and takes the player's
				// own control bar off the bottom edge with it. RESIZE_MODE_FIT
				// letterboxes anything narrower inside the box.
				else -> surface(Modifier.fillMaxWidth().aspectRatio(aspect.coerceIn(1f, 2.4f)))
			}
			Caption(
				item,
				quality = when (val s = streams) {
					is PlaybackStreams.Split -> s.label
					is PlaybackStreams.Single -> s.label
					null -> null
				},
			)
		}
	}
}

/** Matches `show_timeout` in the player layout, so both fade together. */
private const val CONTROLS_TIMEOUT_MS = 3_000L

/**
 * The player's video surface, hosted wherever it is needed.
 *
 * Shared by the expanded player and the docked bar so that moving between them
 * is a reparent rather than a rebuild.
 */
@OptIn(UnstableApi::class)
@Composable
fun NativeSurface(
	playback: NativePlayback,
	useController: Boolean,
	modifier: Modifier = Modifier,
	onControlsVisible: (Boolean) -> Unit = {},
	onFullscreenClick: () -> Unit = {},
) {
	AndroidView(
		factory = { context ->
			LayoutInflater.from(context).inflate(R.layout.native_player_view, null) as PlayerView
		},
		// Re-attaching every time matters: the view is reused across videos, and
		// leaving a stale player on it is what produced audio with a black screen.
		update = { view ->
			if (view.player !== playback.player) view.player = playback.player
			view.useController = useController
			view.setControllerVisibilityListener(
				PlayerView.ControllerVisibilityListener { onControlsVisible(it == View.VISIBLE) }
			)
			view.setFullscreenButtonClickListener { onFullscreenClick() }
		},
		onRelease = { it.player = null },
		modifier = modifier,
	)
}

@Composable
private fun Caption(item: FeedItem, quality: String?, modifier: Modifier = Modifier) {
	Column(modifier.padding(start = 16.dp, end = 16.dp, top = 14.dp, bottom = 20.dp)) {
		Text(item.title, style = MaterialTheme.typography.titleMedium, color = Bright)
		Row {
			if (item.author.isNotEmpty()) {
				Text(item.author, style = MaterialTheme.typography.bodySmall, color = Muted)
			}
			if (!quality.isNullOrEmpty()) {
				Text(
					"  ·  $quality",
					style = MaterialTheme.typography.bodySmall,
					color = Accent.copy(alpha = 0.8f),
				)
			}
		}
	}
}
