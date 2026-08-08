package ink.luciferian.nutube

import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import ink.luciferian.nutube.ui.FeedViewModel
import ink.luciferian.nutube.ui.NuTubeScreen

class MainActivity : ComponentActivity() {

	private val model: FeedViewModel by viewModels()

	override fun onCreate(savedInstanceState: Bundle?) {
		super.onCreate(savedInstanceState)
		enableEdgeToEdge()
		handleIntent(intent)
		setContent {
			MaterialTheme(colorScheme = darkColorScheme()) {
				NuTubeScreen(model)
			}
		}
	}

	override fun onNewIntent(intent: Intent) {
		super.onNewIntent(intent)
		handleIntent(intent)
	}

	/** A shared or opened YouTube link goes straight into the local index. */
	private fun handleIntent(intent: Intent?) {
		val url = when (intent?.action) {
			Intent.ACTION_SEND -> intent.getStringExtra(Intent.EXTRA_TEXT)
			Intent.ACTION_VIEW -> intent.dataString
			else -> null
		} ?: return
		model.indexUrl(url)
	}
}
