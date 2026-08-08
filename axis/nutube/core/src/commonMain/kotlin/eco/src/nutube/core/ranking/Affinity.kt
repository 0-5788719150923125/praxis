package eco.src.nutube.core.ranking

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import okio.FileSystem
import okio.Path

/**
 * What the device has learned from watching, as plain counts.
 *
 * Deliberately legible: two maps of integers, nothing latent. A person can read
 * this file and see exactly why the feed looks the way it does, which is the
 * whole premise - the algorithm lives here and is inspectable.
 */
@Serializable
data class Affinity(
	/** Channel name to number of videos opened from it. */
	val channels: Map<String, Int> = emptyMap(),
	/** Title n-gram to number of opened videos containing it. */
	val titleTerms: Map<String, Int> = emptyMap(),
) {
	val channelPeak: Int get() = channels.values.maxOrNull() ?: 0
	val termPeak: Int get() = titleTerms.values.maxOrNull() ?: 0
}

class AffinityStore(
	private val path: Path,
	private val fs: FileSystem = FileSystem.SYSTEM,
) {

	private val json = Json { ignoreUnknownKeys = true }
	private val mutex = Mutex()
	private val _affinity = MutableStateFlow(Affinity())
	val affinity: StateFlow<Affinity> = _affinity.asStateFlow()

	suspend fun load() = withContext(Dispatchers.Default) {
		if (!fs.exists(path)) return@withContext
		runCatching { json.decodeFromString<Affinity>(fs.read(path) { readUtf8() }) }
			.onSuccess { _affinity.value = it }
		Unit
	}

	/**
	 * Opening a video is the signal. Not a like, not a subscribe - the thing the
	 * user actually did.
	 */
	suspend fun recordOpen(channel: String, title: String) {
		mutex.withLock {
			val current = _affinity.value
			val channels = current.channels.toMutableMap()
			if (channel.isNotBlank()) {
				channels[channel] = (channels[channel] ?: 0) + 1
			}
			val terms = current.titleTerms.toMutableMap()
			// Distinct per video, so one title repeating a word does not stack.
			Tokens.ngrams(title).toSet().forEach { terms[it] = (terms[it] ?: 0) + 1 }
			_affinity.value = current.copy(channels = channels, titleTerms = terms)
		}
		save()
	}

	suspend fun forgetChannel(channel: String) {
		mutex.withLock {
			_affinity.value = _affinity.value.copy(
				channels = _affinity.value.channels - channel
			)
		}
		save()
	}

	suspend fun clear() {
		mutex.withLock { _affinity.value = Affinity() }
		save()
	}

	private suspend fun save() = withContext(Dispatchers.Default) {
		runCatching {
			path.parent?.let { fs.createDirectories(it) }
			fs.write(path) { writeUtf8(json.encodeToString(_affinity.value)) }
		}
		Unit
	}
}
