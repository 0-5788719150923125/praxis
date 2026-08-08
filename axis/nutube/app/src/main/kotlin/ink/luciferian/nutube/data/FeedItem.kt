package ink.luciferian.nutube.data

import kotlinx.serialization.Serializable

/**
 * One indexed video, from any source.
 *
 * Deliberately source-agnostic, the same shape the Godot prototype used: a source
 * plugs in behind this type and the ranker never learns which platform it came from.
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
	/** Populated by the ranker, not persisted as truth. */
	val reason: String = "",
)
