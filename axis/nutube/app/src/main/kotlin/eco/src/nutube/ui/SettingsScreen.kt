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
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import eco.src.nutube.core.PlaybackMode
import eco.src.nutube.core.VideoSource

/**
 * What YouTube's API Services Terms require an API Client to show, and an honest
 * account of which half of this app is one.
 *
 * Embed mode uses the IFrame Player API, so it is an API Client under section 1
 * and the terms genuinely apply to it. Section 11 forbids removing or obscuring
 * links to those terms, and section 7 requires a published privacy policy - both
 * are satisfied here. Native mode uses no YouTube API at all, so it is not
 * covered by this agreement, and the card says so rather than implying the
 * whole app is blessed.
 */
@Composable
private fun ComplianceCard() {
	val uris = LocalUriHandler.current
	Card(
		shape = RoundedCornerShape(14.dp),
		colors = CardDefaults.cardColors(containerColor = Surface),
		modifier = Modifier.fillMaxWidth(),
	) {
		Column(Modifier.padding(16.dp)) {
			Text("Playback and terms", style = MaterialTheme.typography.titleLarge, color = Bright)
			Text(
				"Embedded playback uses YouTube's official player through the IFrame " +
					"Player API, under YouTube's API Services Terms of Service. Its " +
					"advertisements run, its views count and its creators are paid.",
				style = MaterialTheme.typography.bodySmall,
				color = Muted,
				modifier = Modifier.padding(top = 8.dp),
			)
			Text(
				"Native playback does not use YouTube's API and is not covered by " +
					"those terms. It is off by default and turning it on is your choice.",
				style = MaterialTheme.typography.bodySmall,
				color = Muted,
				modifier = Modifier.padding(top = 8.dp),
			)
			Text(
				"nuTube keeps your index, your search terms and what it learns from " +
					"them on this device. It has no account, no analytics and no server " +
					"to send them to.",
				style = MaterialTheme.typography.bodySmall,
				color = Muted,
				modifier = Modifier.padding(top = 8.dp),
			)

			LegalLink("YouTube Terms of Service", "https://www.youtube.com/t/terms", uris::openUri)
			LegalLink("Google Privacy Policy", "https://policies.google.com/privacy", uris::openUri)
			LegalLink(
				"YouTube API Services Terms",
				"https://developers.google.com/youtube/terms/api-services-terms-of-service",
				uris::openUri,
			)
		}
	}
}

@Composable
private fun LegalLink(label: String, url: String, open: (String) -> Unit) {
	Text(
		text = label,
		style = MaterialTheme.typography.bodySmall,
		color = Accent,
		textDecoration = TextDecoration.Underline,
		modifier = Modifier
			.padding(top = 12.dp)
			.clickable { open(url) },
	)
}

/**
 * Per-platform preferences.
 *
 * The only choice today is the playback mode, and it is written to be understood
 * rather than skimmed: the copy states plainly what turning it on takes away
 * from the creator, because that is the whole substance of the setting.
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
		item { ComplianceCard() }

		items(sources, key = { it.id }) { source ->
			val native = modeFor(source.id) == PlaybackMode.NATIVE
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
								onModeChange(
									source.id,
									if (it) PlaybackMode.NATIVE else PlaybackMode.EMBED,
								)
							},
							colors = SwitchDefaults.colors(
								checkedThumbColor = Bright,
								checkedTrackColor = Accent,
								uncheckedTrackColor = SurfaceHigh,
							),
						)
					}

					val uris = LocalUriHandler.current
					LegalLink(
						if (native) "Native playback is outside YouTube's terms - read them"
						else "Playing under YouTube's Terms of Service",
						"https://www.youtube.com/t/terms",
						uris::openUri,
					)
				}
			}
		}
	}
}
