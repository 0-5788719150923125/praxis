package eco.src.nutube.core

import kotlinx.serialization.Serializable

/**
 * One indexed video, from any platform.
 *
 * Deliberately source-agnostic, the same shape the Godot prototype used: a source
 * plugs in behind this type and the ranker never learns which platform it came from.
 * [source] carries the [VideoSource.id] that produced it, so playback can be routed
 * back to the right implementation.
 */
@Serializable
data class FeedItem(
	val id: String,
	val source: String,
	val url: String,
	val title: String = "",
	val author: String = "",
	val thumbnailUrl: String = "",
	val durationSeconds: Long = 0,
	val viewCount: Long = -1,
	val tags: List<String> = emptyList(),
	/**
	 * Which saved search terms surfaced this item. An item can be reached by
	 * several terms, so dropping one term must not delete anything still held by
	 * another - see [LocalIndex.removeTerm].
	 */
	val terms: List<String> = emptyList(),
	/** Added by hand (a shared link), so no term owns it and no term can evict it. */
	val manual: Boolean = false,
	/** Populated by the ranker, not persisted as truth. */
	val reason: String = "",
) {
	/** Stable across platforms, since ids are only unique within a source. */
	val key: String get() = "$source:$id"
}
