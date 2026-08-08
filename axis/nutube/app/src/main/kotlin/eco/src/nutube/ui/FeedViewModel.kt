package eco.src.nutube.ui

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import eco.src.nutube.NuTubeApp
import eco.src.nutube.core.FeedItem
import eco.src.nutube.core.SourceRegistry
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

class FeedViewModel(app: Application) : AndroidViewModel(app) {

	private val index = (app as NuTubeApp).index

	private val _feed = MutableStateFlow<List<FeedItem>>(emptyList())
	val feed: StateFlow<List<FeedItem>> = _feed.asStateFlow()

	private val _busy = MutableStateFlow(false)
	val busy: StateFlow<Boolean> = _busy.asStateFlow()

	private val _error = MutableStateFlow<String?>(null)
	val error: StateFlow<String?> = _error.asStateFlow()

	private var query: String = ""

	init {
		viewModelScope.launch {
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
	}

	/**
	 * Explicit network step: ask every registered platform for the current query
	 * and fold the results into the index. Local ranking still decides the order.
	 */
	fun discover() {
		val q = query.trim()
		if (q.isEmpty() || _busy.value) return
		viewModelScope.launch {
			_busy.value = true
			val found = SourceRegistry.searchAll(q)
			if (found.isEmpty()) _error.value = "nothing came back for \"$q\""
			else { index.upsertAll(found); rerank() }
			_busy.value = false
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

	fun clearError() { _error.value = null }
}
