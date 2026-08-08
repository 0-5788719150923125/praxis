package eco.src.nutube.ui

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import eco.src.nutube.NuTubeApp
import eco.src.nutube.core.FeedItem
import eco.src.nutube.core.PlaybackMode
import eco.src.nutube.core.SourceRegistry
import eco.src.nutube.core.VideoSource
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

class FeedViewModel(app: Application) : AndroidViewModel(app) {

	private val index = (app as NuTubeApp).index
	private val bank = (app as NuTubeApp).terms
	private val settings = (app as NuTubeApp).settings

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
		_feed.value = index.recommend(query)
		_revision.value++
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

	fun indexUrl(url: String) {
		viewModelScope.launch {
			_busy.value = true
			SourceRegistry.resolve(url)
				.onSuccess { index.upsert(it); rerank() }
				.onFailure { _error.value = it.message ?: "could not resolve link" }
			_busy.value = false
		}
	}

	fun clearError() { _error.value = null }
}
