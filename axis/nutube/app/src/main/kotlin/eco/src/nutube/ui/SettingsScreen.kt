package eco.src.nutube.ui

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
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import eco.src.nutube.core.PlaybackMode
import eco.src.nutube.core.VideoSource

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
				}
			}
		}
	}
}
