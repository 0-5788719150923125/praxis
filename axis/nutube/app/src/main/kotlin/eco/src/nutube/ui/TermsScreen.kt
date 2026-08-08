package eco.src.nutube.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

/**
 * The bank of saved searches - nuTube's answer to a subscription list.
 *
 * Every search the user runs lands here and stays until they remove it. Removing
 * a term also clears whatever only that term was holding; anything a surviving
 * term still reaches, or anything added by hand, stays put.
 */
@Composable
fun TermsScreen(
	terms: List<String>,
	exclusiveCount: (String) -> Int,
	onRefresh: (String) -> Unit,
	onRemove: (String) -> Unit,
	contentPadding: PaddingValues,
) {
	if (terms.isEmpty()) {
		Column(
			Modifier.fillMaxSize().padding(contentPadding).padding(32.dp),
			verticalArrangement = Arrangement.Center,
			horizontalAlignment = Alignment.CenterHorizontally,
		) {
			Text("No searches yet", style = MaterialTheme.typography.titleMedium, color = Bright)
			Text(
				"Search from the feed and the term is saved here. The index is built from these.",
				style = MaterialTheme.typography.bodySmall,
				color = Muted,
				modifier = Modifier.padding(top = 8.dp),
			)
		}
		return
	}

	LazyColumn(
		modifier = Modifier.fillMaxSize(),
		contentPadding = PaddingValues(
			start = 16.dp,
			end = 16.dp,
			top = contentPadding.calculateTopPadding() + 12.dp,
			bottom = contentPadding.calculateBottomPadding() + 24.dp,
		),
		verticalArrangement = Arrangement.spacedBy(10.dp),
	) {
		items(terms, key = { it }) { term ->
			Card(
				shape = RoundedCornerShape(12.dp),
				colors = CardDefaults.cardColors(containerColor = Surface),
				modifier = Modifier.fillMaxWidth().clickable { onRefresh(term) },
			) {
				Row(
					Modifier.padding(start = 16.dp, end = 6.dp, top = 6.dp, bottom = 6.dp),
					verticalAlignment = Alignment.CenterVertically,
				) {
					Column(Modifier.weight(1f).padding(vertical = 8.dp)) {
						Text(term, style = MaterialTheme.typography.titleMedium, color = Bright)
						val owned = exclusiveCount(term)
						Text(
							if (owned == 0) "nothing depends on this alone"
							else "$owned video${if (owned == 1) "" else "s"} only this term holds",
							style = MaterialTheme.typography.labelSmall,
							color = Muted,
							modifier = Modifier.padding(top = 2.dp),
						)
					}
					IconButton(onClick = { onRefresh(term) }) {
						Icon(
							Icons.Filled.Refresh,
							contentDescription = "Search \"$term\" again",
							tint = Accent,
							modifier = Modifier.size(20.dp),
						)
					}
					IconButton(onClick = { onRemove(term) }) {
						Icon(
							Icons.Filled.Close,
							contentDescription = "Remove \"$term\"",
							tint = Muted,
							modifier = Modifier.size(20.dp),
						)
					}
				}
			}
		}
	}
}
