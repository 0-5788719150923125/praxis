package eco.src.nutube.ui

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import eco.src.nutube.NuTubeApp
import eco.src.nutube.core.FeedItem
import eco.src.nutube.core.PlaybackMode
import eco.src.nutube.core.Query
import eco.src.nutube.core.SourceRegistry
import eco.src.nutube.core.VideoSource
import eco.src.nutube.core.ranking.AffinityStore
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

class FeedViewModel(app: Application) : AndroidViewModel(app) {

	private val index = (app as NuTubeApp).index
	private val bank = (app as NuTubeApp).terms
	private val settings = (app as NuTubeApp).settings
	private val affinity: AffinityStore = (app as NuTubeApp).affinity

	/** Platforms currently plugged in, for the settings list. */
	val sources: List<VideoSource> get() = SourceRegistry.all()

	/** Re-read on every settings write so the UI recomposes. */
	val playbackModes: StateFlow<Map<String, eco.src.nutube.core.SourceSettings>> = settings.bySource

	/**
	 * True while a native video is on screen. The Activity reads this to decide
	 * whether leaving the app should go to picture-in-picture; the embedded
	 * player is deliberately excluded, since the page owns its own playback.
	 */
	private val _nativePlaybackActive = MutableStateFlow(false)
	val nativePlaybackActive: StateFlow<Boolean> = _nativePlaybackActive.asStateFlow()

	fun setNativePlaybackActive(active: Boolean) { _nativePlaybackActive.value = active }

	/** Outlives the player composables, so picture-in-picture keeps its position. */
	val playback = NativePlayback(app)

	fun stopNativePlayback() = playback.stop()

	override fun onCleared() {
		super.onCleared()
		playback.release()
	}

	fun playbackMode(sourceId: String): PlaybackMode = settings.playbackMode(sourceId)

	fun setPlaybackMode(sourceId: String, mode: PlaybackMode) {
		viewModelScope.launch { settings.setPlaybackMode(sourceId, mode) }
	}

	private val _feed = MutableStateFlow<List<FeedItem>>(emptyList())
	val feed: StateFlow<List<FeedItem>> = _feed.asStateFlow()

	/** The saved search terms, newest first. This is the subscription list. */
	val terms: StateFlow<List<String>> = bank.terms

	private val _busy = MutableStateFlow(false)
	val busy: StateFlow<Boolean> = _busy.asStateFlow()

	private val _error = MutableStateFlow<String?>(null)
	val error: StateFlow<String?> = _error.asStateFlow()

	/**
	 * Bumped every time the ranking is recomputed, so the feed can jump back to
	 * the top. Without it a search silently re-ranks below the current scroll
	 * position and reads as though it did nothing.
	 */
	private val _revision = MutableStateFlow(0)
	val revision: StateFlow<Int> = _revision.asStateFlow()

	private var query: String = ""

	init {
		viewModelScope.launch {
			settings.load()
			affinity.load()
			bank.load()
			index.load()
			rerank()
		}
	}

	/** Rank what is already on the device. No network, no remote algorithm. */
	fun onQueryChanged(text: String) {
		query = text
		rerank()
	}

	private fun rerank() {
		_feed.value = index.recommend(query, affinity.affinity.value)
		_revision.value++
	}

	/**
	 * Opening a video is the only training signal there is. Recorded here, read
	 * back by the ranking rules on the next re-rank.
	 */
	fun recordOpen(item: FeedItem) {
		viewModelScope.launch {
			affinity.recordOpen(item.author, item.title)
			rerank()
		}
	}

	/**
	 * Run the current query against every registered platform, save the term, and
	 * fold the results into the index crediting that term.
	 *
	 * Saving the term is the point: the bank of terms is what a crawler will
	 * eventually re-run on a schedule to keep the index fresh.
	 */
	fun discover() {
		val q = query.trim()
		if (q.isEmpty() || _busy.value) return
		viewModelScope.launch {
			_busy.value = true
			val term = bank.add(q)
			val found = SourceRegistry.searchAll(q)
			if (found.isEmpty()) _error.value = "nothing came back for \"$q\""
			else { index.upsertAll(found, term = term); rerank() }
			_busy.value = false
		}
	}

	/** Re-run a saved term, refreshing whatever it holds. */
	fun refreshTerm(term: String) {
		if (_busy.value) return
		viewModelScope.launch {
			_busy.value = true
			val found = SourceRegistry.searchAll(term)
			if (found.isNotEmpty()) { index.upsertAll(found, term = bank.normalise(term)); rerank() }
			_busy.value = false
		}
	}

	/** Drop a term and everything only it was holding. */
	fun removeTerm(term: String) {
		viewModelScope.launch {
			bank.remove(term)
			index.removeTerm(bank.normalise(term))
			rerank()
		}
	}

	/** How many items would be lost if [term] were removed right now. */
	fun exclusiveCount(term: String): Int = index.countOwnedBy(bank.normalise(term))

	/**
	 * The saved term that follows this item's channel, or null.
	 *
	 * Comparing term strings is not enough: `channel:@the-arc` and
	 * `channel:https://www.youtube.com/channel/UC...` name the same channel and
	 * neither is wrong, so identity is checked against every form the channel is
	 * known by. An item that the channel pull itself brought in already carries
	 * the term, which settles it without any comparison at all.
	 */
	private fun followedTerm(item: FeedItem): String? {
		item.terms.firstOrNull { Query.channelOf(it) != null }?.let { return it }

		val known = listOfNotNull(
			item.authorUrl.takeIf { it.isNotBlank() },
			Query.handleFrom(item.authorUrl),
			item.author.takeIf { it.isNotBlank() },
		)
		if (known.isEmpty()) return null
		return bank.terms.value.firstOrNull { term ->
			val id = Query.channelOf(term) ?: return@firstOrNull false
			known.any { it.equals(id, ignoreCase = true) }
		}
	}

	fun isFollowing(item: FeedItem): Boolean = followedTerm(item) != null

	/**
	 * Following is just a saved term. There is no separate subscription list -
	 * `channel: <name>` goes in the same bank as every other search, shows up in
	 * the Terms tab, and is dropped the same way.
	 */
	fun toggleFollow(item: FeedItem) {
		if (item.author.isBlank() || _busy.value) return
		val existing = followedTerm(item)
		val term = existing ?: Query.channelTerm(item.author, item.authorUrl)
		viewModelScope.launch {
			if (existing != null) {
				bank.remove(existing)
				index.removeTerm(bank.normalise(existing))
			} else {
				_busy.value = true
				try {
					val saved = bank.add(term)
					val found = SourceRegistry.searchAll(term)
					if (found.isEmpty()) _error.value = "no videos found for ${item.author}"
					else index.upsertAll(found, term = saved)
				} finally {
					// Without this a single failure leaves the flag set and every
					// later follow returns silently at the busy guard.
					_busy.value = false
				}
			}
			rerank()
		}
	}

	fun indexUrl(url: String) {
		viewModelScope.launch {
			_busy.value = true
			SourceRegistry.resolve(url)
				.onSuccess { index.upsert(it); rerank() }
				.onFailure { _error.value = it.message ?: "could not resolve link" }
			_busy.value = false
		}
	}

	fun isChannelTerm(term: String): Boolean = Query.channelOf(term) != null

	/**
	 * What a term should read as.
	 *
	 * Channels are always shown by their name, whichever way they were followed.
	 * A typed `@handle` and a tapped Follow store different identifiers - a handle
	 * and a channel URL, because those are what each path can name exactly - but
	 * showing both spellings made one subscription look like two kinds of thing.
	 *
	 * The name is found through the videos the term itself pulled in, which works
	 * identically for both paths since every item carries the term that surfaced
	 * it. The stored identifier is only ever a fallback for a term that has not
	 * yet matched anything.
	 */
	fun termLabel(term: String): String {
		val id = Query.channelOf(term) ?: return term
		val key = Query.canonical(term)
		val named = index.items.value
			.firstOrNull { item ->
				item.author.isNotBlank() &&
					item.terms.any { Query.canonical(it) == key }
			}
			?.author
		if (named != null) return named
		if (id.startsWith("@")) return id
		// Nothing indexed yet: a bare id still reads better than a whole URL.
		return id.substringAfterLast('/').ifBlank { id }
	}

	fun clearError() { _error.value = null }
}
