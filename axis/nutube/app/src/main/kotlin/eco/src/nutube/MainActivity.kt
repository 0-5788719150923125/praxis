package eco.src.nutube

import android.app.PictureInPictureParams
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.os.Bundle
import android.util.Rational
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import eco.src.nutube.ui.FeedViewModel
import eco.src.nutube.ui.NuTubeScreen
import eco.src.nutube.ui.NuTubeTheme

class MainActivity : ComponentActivity() {

	private val model: FeedViewModel by viewModels()

	/** Drives the UI down to just the video surface while in picture-in-picture. */
	private var inPip by mutableStateOf(false)

	private val pipSupported: Boolean
		get() = packageManager.hasSystemFeature(PackageManager.FEATURE_PICTURE_IN_PICTURE)

	override fun onCreate(savedInstanceState: Bundle?) {
		super.onCreate(savedInstanceState)
		enableEdgeToEdge()
		handleIntent(intent)
		setContent {
			NuTubeTheme { NuTubeScreen(model, inPip = inPip) }
		}
	}

	/**
	 * Leaving the app while a native video plays shrinks it into a floating
	 * window instead of abandoning it. Only native playback qualifies - the
	 * embedded player is the platform's own page, and it stops on pause.
	 */
	override fun onUserLeaveHint() {
		super.onUserLeaveHint()
		if (model.nativePlaybackActive.value && pipSupported) {
			runCatching {
				enterPictureInPictureMode(
					PictureInPictureParams.Builder()
						.setAspectRatio(pipAspect())
						.build()
				)
			}
		}
	}

	/**
	 * The floating window takes the video's own shape, so a 4:3 or portrait clip
	 * is not pillarboxed inside a 16:9 frame. Android rejects ratios beyond about
	 * 2.39:1 either way, so clamp before asking.
	 */
	private fun pipAspect(): Rational {
		val a = model.playback.aspect.value.coerceIn(0.42f, 2.38f)
		return Rational((a * 1000).toInt(), 1000)
	}

	override fun onPictureInPictureModeChanged(isInPip: Boolean, config: Configuration) {
		super.onPictureInPictureModeChanged(isInPip, config)
		inPip = isInPip
	}

	override fun onNewIntent(intent: Intent) {
		super.onNewIntent(intent)
		handleIntent(intent)
	}

	/** A shared or opened link goes straight into the local index. */
	private fun handleIntent(intent: Intent?) {
		val url = when (intent?.action) {
			Intent.ACTION_SEND -> intent.getStringExtra(Intent.EXTRA_TEXT)
			Intent.ACTION_VIEW -> intent.dataString
			else -> null
		} ?: return
		model.indexUrl(url)
	}
}
