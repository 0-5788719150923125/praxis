package eco.src.nutube.ui

import android.annotation.SuppressLint
import android.util.Log
import android.view.View
import android.view.ViewGroup
import android.webkit.ConsoleMessage
import android.webkit.WebChromeClient
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.FrameLayout
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner

/**
 * The platform's own embedded player, unmodified.
 *
 * This is the honest mode. It hosts the site's sanctioned embed and gets out of
 * the way: the platform's player runs, its ads run, its view is counted, and the
 * creator is credited. Nothing here reads, rewrites or blocks anything the page
 * loads - if it did, the mode would not be worth having. There is deliberately
 * no JS bridge and no `enablejsapi`; the app does not need to drive this player,
 * and not asking for control is the point.
 *
 * The embed is wrapped in a real `<iframe>` inside a page served from an https
 * base URL rather than navigated to directly. YouTube's `/embed/` endpoint is
 * built to be framed, and loading it as a top-level document with no referrer
 * fails with "Error 153: Video player configuration error".
 */
@SuppressLint("SetJavaScriptEnabled")
@Composable
fun EmbedPlayer(
	url: String,
	modifier: Modifier = Modifier,
	onFullscreenChange: (Boolean) -> Unit = {},
) {
	val lifecycle = LocalLifecycleOwner.current.lifecycle
	val webRef = remember { mutableStateOf<WebView?>(null) }
	val fullscreenView = remember { mutableStateOf<View?>(null) }

	Box(modifier) {
		AndroidView(
			modifier = Modifier.fillMaxSize(),
			factory = { context ->
				FrameLayout(context).also { host ->
					val web = WebView(context).apply {
						settings.javaScriptEnabled = true
						settings.domStorageEnabled = true
						// The user tapped a card to get here, but the WebView cannot
						// see that as a gesture of its own.
						settings.mediaPlaybackRequiresUserGesture = false
						setBackgroundColor(android.graphics.Color.BLACK)
						layoutParams = FrameLayout.LayoutParams(
							ViewGroup.LayoutParams.MATCH_PARENT,
							ViewGroup.LayoutParams.MATCH_PARENT,
						)
						webViewClient = WebViewClient()
						webChromeClient = object : WebChromeClient() {
							override fun onShowCustomView(view: View, cb: CustomViewCallback) {
								// The page's own fullscreen button. We cannot see the
								// player's controls without a JS bridge we deliberately
								// do not have, so this is the only signal there is - and
								// it is the one that matters for the system bars.
								onFullscreenChange(true)
								fullscreenView.value = view
								host.addView(
									view,
									FrameLayout.LayoutParams(
										ViewGroup.LayoutParams.MATCH_PARENT,
										ViewGroup.LayoutParams.MATCH_PARENT,
									),
								)
							}

							override fun onHideCustomView() {
								onFullscreenChange(false)
								fullscreenView.value?.let { host.removeView(it) }
								fullscreenView.value = null
							}

							// The player reports its failures here, so surface them
							// instead of leaving a silent black rectangle.
							override fun onConsoleMessage(m: ConsoleMessage): Boolean {
								Log.d("NuTubeEmbed", "${m.messageLevel()}: ${m.message()}")
								return true
							}
						}
					}
					webRef.value = web
					host.addView(web)
				}
			},
			update = { _ ->
				val web = webRef.value ?: return@AndroidView
				if (web.tag != url) {
					web.tag = url
					web.loadDataWithBaseURL(EMBED_BASE, framePage(url), "text/html", "utf-8", null)
				}
			},
		)
	}

	// Leaving the app stops the embed. There is deliberately no picture-in-picture
	// here: the page owns its own player, and reaching in to drive it would undo
	// the reason this mode exists.
	DisposableEffect(lifecycle) {
		val observer = LifecycleEventObserver { _, event ->
			when (event) {
				Lifecycle.Event.ON_PAUSE -> webRef.value?.onPause()
				Lifecycle.Event.ON_RESUME -> webRef.value?.onResume()
				else -> Unit
			}
		}
		lifecycle.addObserver(observer)
		onDispose {
			lifecycle.removeObserver(observer)
			webRef.value?.apply {
				stopLoading()
				loadUrl("about:blank")
				destroy()
			}
			webRef.value = null
		}
	}
}

/**
 * The origin the embed is framed from. YouTube allows embedding on any site, so
 * this identifies the app rather than impersonating the platform.
 */
private const val EMBED_BASE = "https://nutube.local"

private fun framePage(src: String) = """
<!doctype html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no">
<style>
html,body{margin:0;padding:0;height:100%;background:#000;overflow:hidden}
iframe{border:0;display:block;width:100%;height:100%}
</style>
</head>
<body>
<iframe src="$src"
        allow="autoplay; encrypted-media; picture-in-picture; fullscreen"
        allowfullscreen></iframe>
</body>
</html>
""".trimIndent()
