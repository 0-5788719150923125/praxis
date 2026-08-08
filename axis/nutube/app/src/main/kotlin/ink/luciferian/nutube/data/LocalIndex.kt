package ink.luciferian.nutube.data

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import java.io.File

/**
 * The on-device index and recommender - the whole point of nuTube.
 *
 * There is no remote ranking service. Items live in a JSON file next to the app and
 * are ranked here by inspectable local rules, so every card can say why it surfaced.
 * The scorer is keyword overlap for now, ported straight from the Godot prototype;
 * it is the part meant to grow.
 */
class LocalIndex(private val file: File) {

	private val json = Json { ignoreUnknownKeys = true; prettyPrint = false }
	private val mutex = Mutex()
	private val _items = MutableStateFlow<List<FeedItem>>(emptyList())
	val items: StateFlow<List<FeedItem>> = _items.asStateFlow()

	suspend fun load() = withContext(Dispatchers.IO) {
		if (!file.exists()) return@withContext
		runCatching { json.decodeFromString<List<FeedItem>>(file.readText()) }
			.onSuccess { _items.value = it }
	}

	/** Add or replace by id, then persist. */
	suspend fun upsert(item: FeedItem) {
		mutex.withLock {
			val next = _items.value.toMutableList()
			val at = next.indexOfFirst { it.id == item.id }
			if (at >= 0) next[at] = item else next += item
			_items.value = next
		}
		save()
	}

	suspend fun upsertAll(incoming: List<FeedItem>) {
		mutex.withLock {
			val byId = _items.value.associateBy { it.id }.toMutableMap()
			incoming.forEach { byId[it.id] = it }
			_items.value = byId.values.toList()
		}
		save()
	}

	fun has(id: String): Boolean = _items.value.any { it.id == id }

	private suspend fun save() = withContext(Dispatchers.IO) {
		// `reason` is a ranking artifact, not stored truth.
		val stripped = _items.value.map { it.copy(reason = "") }
		runCatching { file.writeText(json.encodeToString(stripped)) }
		Unit
	}

	/**
	 * Rank the index against [query] and annotate each result with why it surfaced.
	 * An empty query returns a default slice so the feed is never blank.
	 */
	fun recommend(query: String, limit: Int = 50): List<FeedItem> {
		val terms = tokenize(query)
		val pool = _items.value
		if (terms.isEmpty()) {
			return pool.take(limit).map { it.copy(reason = "From your local index") }
		}
		return pool
			.map { it to score(it, terms) }
			.sortedByDescending { it.second }
			.take(limit)
			.map { (item, s) ->
				item.copy(
					reason = when {
						s <= 0.0 -> "Loosely related"
						else -> "Matches your search by tag overlap"
					}
				)
			}
	}

	private fun score(item: FeedItem, terms: List<String>): Double {
		val haystack = tokenize(item.title + " " + item.author + " " + item.tags.joinToString(" ")).toSet()
		val overlap = terms.count { it in haystack }
		return overlap.toDouble() / terms.size
	}

	private fun tokenize(text: String): List<String> =
		text.lowercase()
			.split(Regex("[^\\p{L}\\p{N}]+"))
			.filter { it.length > 1 }
}
