package ink.luciferian.nutube.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.material3.TopAppBar
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
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Search
import androidx.compose.ui.graphics.Color
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import coil3.compose.AsyncImage
import ink.luciferian.nutube.data.FeedItem

/**
 * Search bar over a feed, with a player overlay - the same shape as the Godot
 * prototype, but the list is virtualized and the "Watch" step plays in-app instead
 * of handing off to YouTube.
 */
@OptIn(ExperimentalMaterial3Api::class)
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
		snackbarHost = { SnackbarHost(snackbar) },
		topBar = {
			TopAppBar(title = {
				TextField(
					value = query,
					onValueChange = { query = it; model.onQueryChanged(it) },
					placeholder = { Text("Search your index") },
					singleLine = true,
					modifier = Modifier.fillMaxWidth(),
					trailingIcon = {
						if (busy) CircularProgressIndicator(Modifier.padding(12.dp), strokeWidth = 2.dp)
						else IconButton(onClick = model::discover) {
							Icon(Icons.Filled.Search, contentDescription = "Find more on YouTube")
						}
					},
					keyboardActions = androidx.compose.foundation.text.KeyboardActions(
						onSearch = { model.discover() }
					),
					keyboardOptions = androidx.compose.foundation.text.KeyboardOptions(
						imeAction = ImeAction.Search
					),
				)
			})
		},
	) { padding ->
		LazyColumn(
			modifier = Modifier.fillMaxSize().padding(padding),
			contentPadding = PaddingValues(12.dp),
			verticalArrangement = Arrangement.spacedBy(12.dp),
		) {
			items(feed, key = { it.id }) { item ->
				FeedCard(item) { playing = item }
			}
		}
	}

	playing?.let { item ->
		PlayerOverlay(item = item, onClose = { playing = null })
	}
}

@Composable
private fun FeedCard(item: FeedItem, onClick: () -> Unit) {
	Card(modifier = Modifier.fillMaxWidth().clickable(onClick = onClick)) {
		Column {
			AsyncImage(
				model = item.thumbnailUrl,
				contentDescription = null,
				contentScale = ContentScale.Crop,
				modifier = Modifier.fillMaxWidth().aspectRatio(16f / 9f),
			)
			Column(Modifier.padding(12.dp)) {
				Text(item.title, style = MaterialTheme.typography.titleMedium, maxLines = 2)
				Row(
					verticalAlignment = Alignment.CenterVertically,
					horizontalArrangement = Arrangement.spacedBy(8.dp),
				) {
					if (item.author.isNotEmpty()) {
						Text(item.author, style = MaterialTheme.typography.bodySmall, color = Color.Gray)
					}
				}
				// The index explains itself: every card says why it surfaced.
				if (item.reason.isNotEmpty()) {
					Text(item.reason, style = MaterialTheme.typography.labelSmall, color = Color.Gray)
				}
			}
		}
	}
}
