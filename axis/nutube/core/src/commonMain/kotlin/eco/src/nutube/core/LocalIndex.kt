package eco.src.nutube.core

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import okio.FileSystem
import okio.Path

/**
 * The on-device index and recommender - the whole point of nuTube.
 *
 * There is no remote ranking service. Items live in a JSON file next to the app
 * and are ranked here by inspectable local rules, so every card can say why it
 * surfaced. The scorer is keyword overlap for now, ported from the Godot
 * prototype; it is the part meant to grow.
 *
 * Platform-free by design - it talks to okio rather than java.io, so it moves to
 * a new target unchanged.
 */
class LocalIndex(
	private val path: Path,
	private val fs: FileSystem = FileSystem.SYSTEM,
) {

	private val json = Json { ignoreUnknownKeys = true }
	private val mutex = Mutex()
	private val _items = MutableStateFlow<List<FeedItem>>(emptyList())
	val items: StateFlow<List<FeedItem>> = _items.asStateFlow()

	suspend fun load() = withContext(Dispatchers.Default) {
		if (!fs.exists(path)) return@withContext
		runCatching {
			val text = fs.read(path) { readUtf8() }
			json.decodeFromString<List<FeedItem>>(text)
		}.onSuccess { _items.value = it }
		Unit
	}

	suspend fun upsert(item: FeedItem) = upsertAll(listOf(item))

	suspend fun upsertAll(incoming: List<FeedItem>) {
		if (incoming.isEmpty()) return
		mutex.withLock {
			val byKey = _items.value.associateByTo(LinkedHashMap()) { it.key }
			incoming.forEach { byKey[it.key] = it }
			_items.value = byKey.values.toList()
		}
		save()
	}

	fun has(key: String): Boolean = _items.value.any { it.key == key }

	private suspend fun save() = withContext(Dispatchers.Default) {
		// `reason` is a ranking artifact, not stored truth.
		val stripped = _items.value.map { it.copy(reason = "") }
		runCatching {
			path.parent?.let { fs.createDirectories(it) }
			fs.write(path) { writeUtf8(json.encodeToString(stripped)) }
		}
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
				item.copy(reason = if (s <= 0.0) "Loosely related" else "Matches your search by tag overlap")
			}
	}

	private fun score(item: FeedItem, terms: List<String>): Double {
		val haystack = tokenize(
			item.title + " " + item.author + " " + item.tags.joinToString(" ")
		).toSet()
		return terms.count { it in haystack }.toDouble() / terms.size
	}

	private fun tokenize(text: String): List<String> =
		text.lowercase()
			.split(Regex("[^\\p{L}\\p{N}]+"))
			.filter { it.length > 1 }
}
