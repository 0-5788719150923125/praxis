package eco.src.nutube.ui

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.interaction.DragInteraction
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
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.List
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.NavigationBarItemDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Surface
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.focus.FocusManager
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.platform.SoftwareKeyboardController
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import coil3.compose.AsyncImage
import eco.src.nutube.R
import eco.src.nutube.core.FeedItem
import eco.src.nutube.core.PlaybackMode

/**
 * Logo and search over a feed of cards, closer to the Godot prototype's look than
 * the first Compose pass was.
 *
 * The search field sits in its own full-width row rather than inside a TopAppBar
 * title slot - the title slot reserves room for navigation and action icons, so a
 * TextField placed there gets squeezed past the right edge.
 */
private enum class Tab(val label: String) { Feed("Feed"), Terms("Terms"), Settings("Settings") }

@Composable
fun NuTubeScreen(model: FeedViewModel, inPip: Boolean = false) {
	val feed by model.feed.collectAsStateWithLifecycle()
	val terms by model.terms.collectAsStateWithLifecycle()
	val busy by model.busy.collectAsStateWithLifecycle()
	// Collected so a settings write recomposes the switch and the player route.
	val modes by model.playbackModes.collectAsStateWithLifecycle()
	val revision by model.revision.collectAsStateWithLifecycle()
	val error by model.error.collectAsStateWithLifecycle()

	var tab by rememberSaveable { mutableStateOf(Tab.Feed) }
	var query by rememberSaveable { mutableStateOf("") }
	var playing by remember { mutableStateOf<FeedItem?>(null) }
	// Docked vs expanded. Backing out of a native video docks it instead of
	// stopping it, so the feed comes back and the video carries on.
	var expanded by remember { mutableStateOf(false) }
	val snackbar = remember { SnackbarHostState() }

	val listState = rememberLazyListState()
	val keyboard = LocalSoftwareKeyboardController.current
	val focus = LocalFocusManager.current

	// New ranking, new order - so go back to where the best results now are. A
	// search that re-ranks below the current scroll position looks like a search
	// that did nothing.
	LaunchedEffect(revision) {
		if (revision > 0 && listState.firstVisibleItemIndex > 0) listState.scrollToItem(0)
	}

	// Dragging the feed is a clear signal the user is done typing; without this the
	// keyboard sits open over half the screen until dismissed by hand. It watches
	// drags specifically rather than any scroll, because the jump-to-top above is
	// also a scroll and would otherwise close the keyboard on every keystroke.
	LaunchedEffect(listState.interactionSource) {
		listState.interactionSource.interactions.collect { interaction ->
			if (interaction is DragInteraction.Start) dismissKeyboard(keyboard, focus)
		}
	}
	LaunchedEffect(tab) { dismissKeyboard(keyboard, focus) }

	val playingMode = playing?.let { modes[it.source]?.playback ?: model.playbackMode(it.source) }
	val dockable = playingMode == PlaybackMode.NATIVE

	fun stopPlayback() {
		playing = null
		expanded = false
		model.stopNativePlayback()
	}

	// The Activity asks this before going to system picture-in-picture, and a
	// docked video is still playing, so presence follows the video and not the
	// expanded overlay.
	LaunchedEffect(playing, dockable) {
		model.setNativePlaybackActive(playing != null && dockable)
	}

	LaunchedEffect(error) {
		error?.let { snackbar.showSnackbar(it); model.clearError() }
	}

	Box(Modifier.fillMaxSize()) {
	if (!inPip) Scaffold(
		containerColor = MaterialTheme.colorScheme.background,
		snackbarHost = { SnackbarHost(snackbar) },
		topBar = {
			Column(Modifier.statusBarsPadding().padding(horizontal = 16.dp)) {
				Brand()
			}
		},
		bottomBar = {
			Column {
				playing?.takeIf { !expanded && dockable }?.let { item ->
					MiniPlayer(
						item = item,
						playback = model.playback,
						onExpand = { expanded = true },
						onClose = { stopPlayback() },
					)
				}
				NavigationBar(containerColor = Surface) {
				Tab.entries.forEach { entry ->
					NavigationBarItem(
						selected = tab == entry,
						onClick = { tab = entry },
						icon = {
							Icon(
								when (entry) {
									Tab.Feed -> Icons.Filled.PlayArrow
									Tab.Terms -> Icons.Filled.List
									Tab.Settings -> Icons.Filled.Settings
								},
								contentDescription = entry.label,
							)
						},
						label = { Text(entry.label) },
						colors = NavigationBarItemDefaults.colors(
							selectedIconColor = Accent,
							selectedTextColor = Accent,
							indicatorColor = SurfaceHigh,
							unselectedIconColor = Muted,
							unselectedTextColor = Muted,
						),
					)
				}
				}
			}
		},
	) { padding ->
		Box(Modifier.fillMaxSize()) {
		when (tab) {
			Tab.Feed -> LazyColumn(
				state = listState,
				modifier = Modifier.fillMaxSize().padding(padding),
				// Room for the floating search to sit over the feed without ever
				// covering the last card.
				contentPadding = PaddingValues(
					start = 16.dp,
					end = 16.dp,
					top = 12.dp,
					bottom = SEARCH_LANE + 24.dp,
				),
				verticalArrangement = Arrangement.spacedBy(14.dp),
			) {
				items(feed, key = { it.key }) { item ->
					FeedCard(item) {
						dismissKeyboard(keyboard, focus)
						// Opening is the signal the ranking rules learn from.
						model.recordOpen(item)
						playing = item
						expanded = true
					}
				}
			}

			Tab.Terms -> Box(Modifier.fillMaxSize()) {
				TermsScreen(
					terms = terms,
					exclusiveCount = model::exclusiveCount,
					label = model::termLabel,
					isChannel = model::isChannelTerm,
					onRefresh = model::refreshTerm,
					onRemove = model::removeTerm,
					contentPadding = padding,
				)
			}

			Tab.Settings -> Box(Modifier.fillMaxSize()) {
				SettingsScreen(
					sources = model.sources,
					modeFor = { id -> modes[id]?.playback ?: model.playbackMode(id) },
					onModeChange = model::setPlaybackMode,
					contentPadding = padding,
				)
			}
		}

		// The search floats over the feed rather than sitting above it: the list
		// scrolls underneath, and the thing you reach for most is the thing
		// closest to your thumb.
		if (tab == Tab.Feed) {
			SearchField(
				value = query,
				busy = busy,
				onValueChange = { query = it; model.onQueryChanged(it) },
				onSubmit = { dismissKeyboard(keyboard, focus); model.discover() },
				modifier = Modifier
					.align(Alignment.BottomCenter)
					// Above the tab bar when idle, above the keyboard when typing -
					// whichever is taller, never both stacked. Summing them (a bottom
					// padding plus imePadding) lifted the pill into the middle of the
					// feed as soon as the keyboard opened.
					.padding(
						start = 16.dp,
						end = 16.dp,
						// Scaffold already shrinks this content area to the top of the
						// keyboard, so the only inset to add is the bar below. Adding
						// the IME height as well - directly, or via imePadding - lifts
						// the pill twice and strands it in the middle of the feed.
						bottom = padding.calculateBottomPadding() + 12.dp,
					),
			)
		}
		}
	}

	playing?.takeIf { expanded || inPip }?.let { item ->
		PlayerOverlay(
			item = item,
			mode = playingMode ?: PlaybackMode.EMBED,
			inPip = inPip,
			playback = model.playback,
			following = model.isFollowing(item),
			onToggleFollow = { model.toggleFollow(item) },
			// Native docks; the embedded player owns its own playback and cannot be
			// carried into a bar, so backing out of it stops.
			onClose = { if (dockable) expanded = false else stopPlayback() },
		)
	}
	}
}

private fun dismissKeyboard(keyboard: SoftwareKeyboardController?, focus: FocusManager) {
	keyboard?.hide()
	// Hiding alone leaves the field focused, so the next tap re-opens the keyboard.
	focus.clearFocus()
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

/** Height the feed reserves so the floating search never hides the last card. */
private val SEARCH_LANE = 84.dp


@Composable
private fun SearchField(
	value: String,
	busy: Boolean,
	onValueChange: (String) -> Unit,
	onSubmit: () -> Unit,
	modifier: Modifier = Modifier,
) {
	Surface(
		shape = RoundedCornerShape(28.dp),
		color = SurfaceHigh,
		shadowElevation = 10.dp,
		modifier = modifier.fillMaxWidth(),
	) {
		TextField(
			value = value,
			onValueChange = onValueChange,
			placeholder = { Text("Search, or @channel", color = Muted) },
			singleLine = true,
			shape = RoundedCornerShape(28.dp),
			colors = TextFieldDefaults.colors(
				focusedContainerColor = SurfaceHigh,
				unfocusedContainerColor = SurfaceHigh,
				focusedIndicatorColor = Color.Transparent,
				unfocusedIndicatorColor = Color.Transparent,
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
			modifier = Modifier.fillMaxWidth(),
		)
	}
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
				val byline = listOfNotNull(
					item.author.takeIf { it.isNotEmpty() },
					uploadedLabel(item).takeIf { it.isNotEmpty() },
				).joinToString("  ·  ")
				if (byline.isNotEmpty()) {
					Text(
						byline,
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
