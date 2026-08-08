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
	private val bank = (app as NuTubeApp).terms

	private val _feed = MutableStateFlow<List<FeedItem>>(emptyList())
	val feed: StateFlow<List<FeedItem>> = _feed.asStateFlow()

	/** The saved search terms, newest first. This is the subscription list. */
	val terms: StateFlow<List<String>> = bank.terms

	private val _busy = MutableStateFlow(false)
	val busy: StateFlow<Boolean> = _busy.asStateFlow()

	private val _error = MutableStateFlow<String?>(null)
	val error: StateFlow<String?> = _error.asStateFlow()

	private var query: String = ""

	init {
		viewModelScope.launch {
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
