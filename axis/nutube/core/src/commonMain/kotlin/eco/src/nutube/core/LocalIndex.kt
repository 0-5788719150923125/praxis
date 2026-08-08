package eco.src.nutube.core

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import eco.src.nutube.core.ranking.Affinity
import eco.src.nutube.core.ranking.RankingContext
import eco.src.nutube.core.ranking.Ranker
import eco.src.nutube.core.ranking.Tokens
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
	private val ranker = Ranker()
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

	/**
	 * Fold [incoming] into the index, crediting [term] for anything it surfaced.
	 *
	 * Existing items keep the terms they already had - the same video can be
	 * reachable from several searches, and every one of them is a reason to keep
	 * it. Items arriving with no term were added by hand and are marked as such.
	 */
	suspend fun upsertAll(incoming: List<FeedItem>, term: String? = null) {
		if (incoming.isEmpty()) return
		mutex.withLock {
			val byKey = _items.value.associateByTo(LinkedHashMap()) { it.key }
			incoming.forEach { fresh ->
				val existing = byKey[fresh.key]
				byKey[fresh.key] = fresh.copy(
					terms = (existing?.terms.orEmpty() + listOfNotNull(term)).distinct(),
					manual = existing?.manual ?: (term == null),
				)
			}
			_items.value = byKey.values.toList()
		}
		save()
	}

	/**
	 * Forget a search term and everything only it was holding.
	 *
	 * An item reachable from another surviving term stays, minus the credit. An
	 * item added by hand stays regardless - no term put it there, so no term can
	 * take it away.
	 */
	suspend fun removeTerm(term: String) {
		mutex.withLock {
			_items.value = _items.value.mapNotNull { item ->
				if (term !in item.terms) return@mapNotNull item
				val remaining = item.terms - term
				if (remaining.isEmpty() && !item.manual) null else item.copy(terms = remaining)
			}
		}
		save()
	}

	/** How many items [term] is currently the only holder of. */
	fun countOwnedBy(term: String): Int =
		_items.value.count { term in it.terms && it.terms.size == 1 && !it.manual }

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
	 * Rank the index and annotate each result with why it surfaced.
	 *
	 * The scoring itself lives in [eco.src.nutube.core.ranking.RULE_REGISTRY];
	 * this only supplies the pool and the context. Adding a signal is a new rule,
	 * not a change here.
	 */
	fun recommend(query: String, affinity: Affinity = Affinity(), limit: Int = 50): List<FeedItem> {
		val ctx = RankingContext(queryTerms = Tokens.words(query), affinity = affinity)
		return ranker.rank(_items.value, ctx, limit)
	}

}
