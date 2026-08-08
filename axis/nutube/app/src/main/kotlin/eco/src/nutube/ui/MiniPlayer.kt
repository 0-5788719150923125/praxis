package eco.src.nutube.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import eco.src.nutube.core.FeedItem

/**
 * The docked player: a video that keeps going while you carry on browsing.
 *
 * Backing out of the expanded player lands here rather than stopping playback,
 * which is the whole point - the feed is the thing you came back for, and the
 * video was not the reason to leave it. The surface moves here from the expanded
 * player without a restart because the player itself lives outside the UI tree.
 */
@Composable
fun MiniPlayer(
	item: FeedItem,
	playback: NativePlayback,
	onExpand: () -> Unit,
	onClose: () -> Unit,
) {
	Surface(color = SurfaceHigh, tonalElevation = 3.dp) {
		Row(
			modifier = Modifier
				.fillMaxWidth()
				.height(64.dp)
				.clickable(onClick = onExpand),
			verticalAlignment = Alignment.CenterVertically,
		) {
			// No controls at this size; the whole bar is one target that expands.
			NativeSurface(
				playback = playback,
				useController = false,
				modifier = Modifier.width(114.dp).height(64.dp),
			)
			Column(
				Modifier.weight(1f).padding(horizontal = 12.dp),
				verticalArrangement = Arrangement.Center,
			) {
				Text(
					item.title,
					style = MaterialTheme.typography.bodySmall,
					color = Bright,
					maxLines = 1,
				)
				if (item.author.isNotEmpty()) {
					Text(
						item.author,
						style = MaterialTheme.typography.labelSmall,
						color = Muted,
						maxLines = 1,
					)
				}
			}
			IconButton(onClick = onClose) {
				Icon(
					Icons.Filled.Close,
					contentDescription = "Stop playback",
					tint = Muted,
					modifier = Modifier.size(20.dp),
				)
			}
		}
	}
}
