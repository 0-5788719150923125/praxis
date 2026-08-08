package eco.src.nutube.core

import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope

/**
 * The set of platforms currently plugged in.
 *
 * Sources register themselves at startup from the platform layer, because their
 * implementations are platform-specific even when this registry is not - YouTube
 * needs a JVM to run the player JavaScript, so it can only register on Android
 * and desktop.
 */
object SourceRegistry {

	private val sources = mutableMapOf<String, VideoSource>()

	fun register(source: VideoSource) {
		sources[source.id] = source
	}

	fun all(): List<VideoSource> = sources.values.toList()

	fun byId(id: String): VideoSource? = sources[id]

	/** The source that claims [url], if any. */
	fun forUrl(url: String): VideoSource? = sources.values.firstOrNull { it.handles(url) }

	/** The source that owns [item], resolved through [FeedItem.source]. */
	fun forItem(item: FeedItem): VideoSource? = sources[item.source]

	/**
	 * Search every registered platform at once and flatten the results.
	 *
	 * A source that fails contributes nothing rather than failing the whole search,
	 * so one platform being down or rate-limited still leaves a usable feed.
	 */
	suspend fun searchAll(query: String, limitPerSource: Int = 20): List<FeedItem> = coroutineScope {
		val parsed = Query.parse(query)
		all()
			.map { source ->
				async {
					if (parsed.isChannel) {
						source.channelVideos(parsed.channel!!).getOrDefault(emptyList())
					} else {
						source.search(parsed.text, limitPerSource).getOrDefault(emptyList())
					}
				}
			}
			.awaitAll()
			.flatten()
	}

	/** Index a URL through whichever source recognises it. */
	suspend fun resolve(url: String): Result<FeedItem> =
		forUrl(url)?.resolve(url)
			?: Result.failure(IllegalArgumentException("no source handles $url"))
}
