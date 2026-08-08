package eco.src.nutube.ui

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import coil3.compose.AsyncImage
import eco.src.nutube.R
import eco.src.nutube.core.FeedItem

/**
 * Logo and search over a feed of cards, closer to the Godot prototype's look than
 * the first Compose pass was.
 *
 * The search field sits in its own full-width row rather than inside a TopAppBar
 * title slot - the title slot reserves room for navigation and action icons, so a
 * TextField placed there gets squeezed past the right edge.
 */
@Composable
fun NuTubeScreen(model: FeedViewModel) {
	val feed by model.feed.collectAsStateWithLifecycle()
	val busy by model.busy.collectAsStateWithLifecycle()
	val error by model.error.collectAsStateWithLifecycle()

	var query by rememberSaveable { mutableStateOf("") }
	var playing by remember { mutableStateOf<FeedItem?>(null) }
	val snackbar = remember { SnackbarHostState() }

	LaunchedEffect(error) {
		error?.let { snackbar.showSnackbar(it); model.clearError() }
	}

	Scaffold(
		containerColor = MaterialTheme.colorScheme.background,
		snackbarHost = { SnackbarHost(snackbar) },
		topBar = {
			Column(Modifier.statusBarsPadding().padding(horizontal = 16.dp)) {
				Brand()
				SearchField(
					value = query,
					busy = busy,
					onValueChange = { query = it; model.onQueryChanged(it) },
					onSubmit = model::discover,
				)
			}
		},
	) { padding ->
		LazyColumn(
			modifier = Modifier.fillMaxSize().padding(padding),
			contentPadding = PaddingValues(start = 16.dp, end = 16.dp, top = 12.dp, bottom = 24.dp),
			verticalArrangement = Arrangement.spacedBy(14.dp),
		) {
			items(feed, key = { it.key }) { item ->
				FeedCard(item) { playing = item }
			}
		}
	}

	playing?.let { item ->
		PlayerOverlay(item = item, onClose = { playing = null })
	}
}

@Composable
private fun Brand() {
	Row(
		modifier = Modifier.fillMaxWidth().padding(top = 12.dp, bottom = 10.dp),
		verticalAlignment = Alignment.CenterVertically,
		horizontalArrangement = Arrangement.spacedBy(10.dp),
	) {
		Image(
			painter = painterResource(R.drawable.logo_nutube),
			contentDescription = null,
			modifier = Modifier.size(28.dp),
		)
		Text("nuTube", style = MaterialTheme.typography.titleLarge, color = Bright)
	}
}

@Composable
private fun SearchField(
	value: String,
	busy: Boolean,
	onValueChange: (String) -> Unit,
	onSubmit: () -> Unit,
) {
	TextField(
		value = value,
		onValueChange = onValueChange,
		placeholder = { Text("Search your index", color = Muted) },
		singleLine = true,
		shape = RoundedCornerShape(12.dp),
		colors = TextFieldDefaults.colors(
			focusedContainerColor = SurfaceHigh,
			unfocusedContainerColor = SurfaceHigh,
			focusedIndicatorColor = androidx.compose.ui.graphics.Color.Transparent,
			unfocusedIndicatorColor = androidx.compose.ui.graphics.Color.Transparent,
		),
		trailingIcon = {
			if (busy) {
				CircularProgressIndicator(Modifier.padding(14.dp).size(18.dp), strokeWidth = 2.dp)
			} else {
				IconButton(onClick = onSubmit) {
					Icon(Icons.Filled.Search, contentDescription = "Find more on YouTube", tint = Accent)
				}
			}
		},
		keyboardOptions = KeyboardOptions(imeAction = ImeAction.Search),
		keyboardActions = KeyboardActions(onSearch = { onSubmit() }),
		// Width comes from the parent's padding, not from a title slot's leftovers.
		modifier = Modifier.fillMaxWidth().padding(bottom = 10.dp),
	)
}

@Composable
private fun FeedCard(item: FeedItem, onClick: () -> Unit) {
	Card(
		modifier = Modifier.fillMaxWidth().clickable(onClick = onClick),
		shape = RoundedCornerShape(14.dp),
		colors = CardDefaults.cardColors(containerColor = Surface),
	) {
		Column {
			Box {
				AsyncImage(
					model = item.thumbnailUrl,
					contentDescription = null,
					contentScale = ContentScale.Crop,
					modifier = Modifier.fillMaxWidth().aspectRatio(16f / 9f),
				)
				if (item.durationSeconds > 0) {
					Text(
						text = formatDuration(item.durationSeconds),
						style = MaterialTheme.typography.labelSmall,
						color = Bright,
						modifier = Modifier
							.align(Alignment.BottomEnd)
							.padding(8.dp)
							.background(Ink.copy(alpha = 0.82f), RoundedCornerShape(4.dp))
							.padding(horizontal = 5.dp, vertical = 2.dp),
					)
				}
			}
			Column(Modifier.padding(horizontal = 14.dp, vertical = 12.dp)) {
				Text(
					item.title,
					style = MaterialTheme.typography.titleMedium,
					color = Bright,
					maxLines = 2,
				)
				if (item.author.isNotEmpty()) {
					Text(
						item.author,
						style = MaterialTheme.typography.bodySmall,
						color = Muted,
						modifier = Modifier.padding(top = 3.dp),
					)
				}
				// The index explains itself: every card says why it surfaced.
				if (item.reason.isNotEmpty()) {
					Text(
						item.reason,
						style = MaterialTheme.typography.labelSmall,
						color = Accent.copy(alpha = 0.75f),
						modifier = Modifier.padding(top = 5.dp),
					)
				}
			}
		}
	}
}

private fun formatDuration(seconds: Long): String {
	val h = seconds / 3600
	val m = (seconds % 3600) / 60
	val s = seconds % 60
	return if (h > 0) "%d:%02d:%02d".format(h, m, s) else "%d:%02d".format(m, s)
}
