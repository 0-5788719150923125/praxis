package eco.src.nutube.core

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

@Serializable
data class SourceSettings(
	val playback: PlaybackMode = PlaybackMode.EMBED,
)

/**
 * Per-platform preferences, keyed by [VideoSource.id].
 *
 * Settings are per-source rather than global because the trade differs by
 * platform: a site with a sanctioned embed and one without are not the same
 * decision, and a future PeerTube source has no ads to skip in the first place.
 */
class Settings(
	private val path: Path,
	private val fs: FileSystem = FileSystem.SYSTEM,
) {

	private val json = Json { ignoreUnknownKeys = true }
	private val mutex = Mutex()
	private val _bySource = MutableStateFlow<Map<String, SourceSettings>>(emptyMap())
	val bySource: StateFlow<Map<String, SourceSettings>> = _bySource.asStateFlow()

	suspend fun load() = withContext(Dispatchers.Default) {
		if (!fs.exists(path)) return@withContext
		runCatching {
			json.decodeFromString<Map<String, SourceSettings>>(fs.read(path) { readUtf8() })
		}.onSuccess { _bySource.value = it }
		Unit
	}

	fun forSource(id: String): SourceSettings = _bySource.value[id] ?: SourceSettings()

	fun playbackMode(id: String): PlaybackMode = forSource(id).playback

	suspend fun setPlaybackMode(id: String, mode: PlaybackMode) {
		mutex.withLock {
			_bySource.value = _bySource.value + (id to forSource(id).copy(playback = mode))
		}
		save()
	}

	private suspend fun save() = withContext(Dispatchers.Default) {
		runCatching {
			path.parent?.let { fs.createDirectories(it) }
			fs.write(path) { writeUtf8(json.encodeToString(_bySource.value)) }
		}
		Unit
	}
}
