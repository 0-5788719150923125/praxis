package eco.src.nutube.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.unit.dp
import eco.src.nutube.core.PlaybackMode
import eco.src.nutube.core.PlatformTerms
import eco.src.nutube.core.VideoSource

/**
 * Per-platform preferences.
 *
 * Everything about a platform - its playback mode, what each mode costs the
 * creator, and the notices it requires - lives in that platform's own card,
 * because each platform sets its own terms. The screen renders whatever a
 * [VideoSource] declares rather than knowing anything about YouTube itself.
 */
@Composable
fun SettingsScreen(
	sources: List<VideoSource>,
	modeFor: (String) -> PlaybackMode,
	onModeChange: (String, PlaybackMode) -> Unit,
	contentPadding: PaddingValues,
) {
	LazyColumn(
		modifier = Modifier.fillMaxSize(),
		contentPadding = PaddingValues(
			start = 16.dp,
			end = 16.dp,
			top = contentPadding.calculateTopPadding() + 12.dp,
			bottom = contentPadding.calculateBottomPadding() + 24.dp,
		),
		verticalArrangement = Arrangement.spacedBy(12.dp),
	) {
		items(sources, key = { it.id }) { source ->
			SourceCard(
				source = source,
				native = modeFor(source.id) == PlaybackMode.NATIVE,
				onModeChange = { onModeChange(source.id, it) },
			)
		}

		// About nuTube rather than any one platform, so it sits outside the cards.
		item { AppPrivacyNote() }
	}
}

@Composable
private fun SourceCard(
	source: VideoSource,
	native: Boolean,
	onModeChange: (PlaybackMode) -> Unit,
) {
	Card(
		shape = RoundedCornerShape(14.dp),
		colors = CardDefaults.cardColors(containerColor = Surface),
		modifier = Modifier.fillMaxWidth(),
	) {
		Column(Modifier.padding(16.dp)) {
			Text(source.displayName, style = MaterialTheme.typography.titleLarge, color = Bright)

			Row(
				Modifier.fillMaxWidth().padding(top = 14.dp),
				verticalAlignment = Alignment.CenterVertically,
			) {
				Column(Modifier.weight(1f).padding(end = 12.dp)) {
					Text(
						"Native playback",
						style = MaterialTheme.typography.titleMedium,
						color = Bright,
					)
					Text(
						if (native)
							"On. Streams play directly, so you get hardware decode, quality " +
								"control and picture-in-picture. No advertisement is shown and " +
								"the view is not counted, so the creator earns nothing from it."
						else
							"Off. Videos play in ${source.displayName}'s own embedded player, " +
								"so its advertisements run, the view is counted and the creator " +
								"is credited exactly as on the site itself.",
						style = MaterialTheme.typography.bodySmall,
						color = Muted,
						modifier = Modifier.padding(top = 4.dp),
					)
				}
				Switch(
					checked = native,
					onCheckedChange = {
						onModeChange(if (it) PlaybackMode.NATIVE else PlaybackMode.EMBED)
					},
					colors = SwitchDefaults.colors(
						checkedThumbColor = Bright,
						checkedTrackColor = Accent,
						uncheckedTrackColor = SurfaceHigh,
					),
				)
			}

			source.terms?.let { TermsBlock(it, native) }
		}
	}
}

/**
 * The platform's own notices.
 *
 * The mode in effect is stated first, since that is the one the user is actually
 * relying on, but both are always shown - the trade is only meaningful if you can
 * see the other half of it. The links are never collapsed behind a control:
 * YouTube's API Services Terms forbid obscuring them, and the same courtesy
 * costs nothing for any other platform.
 */
@Composable
private fun TermsBlock(terms: PlatformTerms, native: Boolean) {
	val uris = LocalUriHandler.current

	HorizontalDivider(
		Modifier.padding(vertical = 14.dp),
		color = MaterialTheme.colorScheme.outline,
	)

	val ordered = if (native) listOf(terms.nativeNote, terms.embedNote)
	else listOf(terms.embedNote, terms.nativeNote)

	ordered.forEachIndexed { i, note ->
		Text(
			note,
			style = MaterialTheme.typography.bodySmall,
			color = if (i == 0) Muted else Muted.copy(alpha = 0.72f),
			modifier = Modifier.padding(bottom = 8.dp),
		)
	}

	terms.links.forEach { link ->
		Text(
			text = link.label,
			style = MaterialTheme.typography.bodySmall,
			color = Accent,
			textDecoration = TextDecoration.Underline,
			modifier = Modifier
				.padding(top = 10.dp)
				.clickable { uris.openUri(link.url) },
		)
	}
}

@Composable
private fun AppPrivacyNote() {
	Text(
		"nuTube keeps your index, your search terms and what it learns from them on " +
			"this device. It has no account, no analytics and no server to send them to.",
		style = MaterialTheme.typography.bodySmall,
		color = Muted,
		modifier = Modifier.padding(horizontal = 4.dp, vertical = 8.dp),
	)
}
