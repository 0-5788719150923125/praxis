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
 * Every search the user has run, kept as the standing description of what they
 * want to see.
 *
 * This is the subscription list. A term earns its place by being searched, and
 * the index is built from the terms rather than from a follow button. A crawler
 * will eventually re-run these on a schedule; for now they are only run when the
 * user searches.
 */
class TermBank(
	private val path: Path,
	private val fs: FileSystem = FileSystem.SYSTEM,
) {

	private val json = Json { ignoreUnknownKeys = true }
	private val mutex = Mutex()
	private val _terms = MutableStateFlow<List<String>>(emptyList())

	/** Most recently searched first. */
	val terms: StateFlow<List<String>> = _terms.asStateFlow()

	suspend fun load() = withContext(Dispatchers.Default) {
		if (!fs.exists(path)) return@withContext
		runCatching {
			json.decodeFromString<List<String>>(fs.read(path) { readUtf8() })
		}.onSuccess { _terms.value = it }
		Unit
	}

	/**
	 * Record a search. Returns the normalised term, or null if it was blank.
	 * Re-searching an existing term moves it to the front rather than duplicating.
	 */
	suspend fun add(raw: String): String? {
		val term = normalise(raw)
		if (term.isEmpty()) return null
		mutex.withLock {
			_terms.value = listOf(term) + _terms.value.filterNot { it == term }
		}
		save()
		return term
	}

	suspend fun remove(term: String) {
		val key = normalise(term)
		mutex.withLock { _terms.value = _terms.value.filterNot { it == key } }
		save()
	}

	/** Collapse case and whitespace so "Lock Picking" and "lock  picking" are one term. */
	fun normalise(raw: String): String = raw.trim().lowercase().replace(Regex("\\s+"), " ")

	private suspend fun save() = withContext(Dispatchers.Default) {
		runCatching {
			path.parent?.let { fs.createDirectories(it) }
			fs.write(path) { writeUtf8(json.encodeToString(_terms.value)) }
		}
		Unit
	}
}
