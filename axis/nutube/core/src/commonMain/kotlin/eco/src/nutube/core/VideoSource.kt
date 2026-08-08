package eco.src.nutube.core

/**
 * A video platform nuTube can pull from.
 *
 * Everything above this interface - the index, the ranker, the UI - is written
 * against [FeedItem] and never learns which platform an item came from. Adding a
 * platform means implementing this and registering it; nothing else changes.
 *
 * Implementations must be safe to call from any coroutine and must not throw:
 * failures come back as a failed [Result] so one broken source cannot take the
 * feed down with it.
 */
interface VideoSource {

	/** Stable key, stored on every [FeedItem.source] so items can be routed back here. */
	val id: String

	/** Shown in the UI when more than one source is active. */
	val displayName: String

	/**
	 * A page for the platform's own embedded player, or null if it has none.
	 *
	 * When present this is the default playback route: the platform serves its
	 * own ads and counts its own view, so the creator is credited exactly as if
	 * the video had been opened on the platform itself.
	 */
	fun embedUrl(itemId: String): String? = null

	/**
	 * This platform's own terms and required notices, rendered in Settings beside
	 * its playback toggle. Null for a platform that asks for nothing.
	 */
	val terms: PlatformTerms? get() = null

	/** True if this source recognises [url] as one of its own. */
	fun handles(url: String): Boolean

	/** Search the platform. Results are not yet in the index; the caller decides. */
	suspend fun search(query: String, limit: Int = 20): Result<List<FeedItem>>

	/** Turn a single URL into an indexable item. */
	suspend fun resolve(url: String): Result<FeedItem>

	/** How to play [itemId], preferring the highest quality the platform offers. */
	suspend fun streams(itemId: String): Result<PlaybackStreams>
}
