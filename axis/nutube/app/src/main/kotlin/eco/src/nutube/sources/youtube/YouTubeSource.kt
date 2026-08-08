package eco.src.nutube.sources.youtube

import eco.src.nutube.core.FeedItem
import eco.src.nutube.core.PlaybackStreams
import eco.src.nutube.core.VideoSource
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.schabi.newpipe.extractor.Image
import org.schabi.newpipe.extractor.ServiceList
import org.schabi.newpipe.extractor.search.SearchInfo
import org.schabi.newpipe.extractor.stream.StreamInfo
import org.schabi.newpipe.extractor.stream.StreamInfoItem

/**
 * YouTube behind the generic [VideoSource] contract, with no API key and no OAuth.
 *
 * NewPipeExtractor speaks YouTube's InnerTube protocol and runs the player's
 * JavaScript through Rhino to decipher stream URLs. That JS step is why this could
 * not have stayed in GDScript, and why it lives in androidMain rather than
 * commonMain - the library is JVM-only.
 */
object YouTubeSource : VideoSource {

	override val id = "youtube"
	override val displayName = "YouTube"

	private val service = ServiceList.YouTube
	private val ID = Regex("(?:v=|youtu\\.be/|/shorts/|/embed/|/live/)([A-Za-z0-9_-]{11})")

	override fun handles(url: String): Boolean =
		("youtube.com" in url || "youtu.be" in url) && ID.containsMatchIn(url)

	fun idFromUrl(url: String): String? = ID.find(url)?.groupValues?.get(1)

	private fun watchUrl(id: String) = "https://www.youtube.com/watch?v=$id"

	override suspend fun search(query: String, limit: Int): Result<List<FeedItem>> =
		withContext(Dispatchers.IO) {
			runCatching {
				val handler = service.searchQHFactory.fromQuery(query, listOf("videos"), "")
				SearchInfo.getInfo(service, handler)
					.relatedItems
					.filterIsInstance<StreamInfoItem>()
					.take(limit)
					.mapNotNull { it.toFeedItem() }
			}
		}

	override suspend fun resolve(url: String): Result<FeedItem> = withContext(Dispatchers.IO) {
		runCatching {
			val videoId = idFromUrl(url) ?: error("not a YouTube video URL: $url")
			StreamInfo.getInfo(service, watchUrl(videoId)).let { info ->
				FeedItem(
					id = videoId,
					source = id,
					url = watchUrl(videoId),
					title = info.name.orEmpty(),
					author = info.uploaderName.orEmpty(),
					thumbnailUrl = info.thumbnails.best(),
					durationSeconds = info.duration,
					viewCount = info.viewCount,
					tags = info.tags.orEmpty(),
				)
			}
		}
	}

	/**
	 * Best available quality.
	 *
	 * YouTube's muxed progressive streams stop at 720p. Anything above that is
	 * published as video-only, with audio as a separate track, so HD means handing
	 * the player both and letting it merge them. The muxed stream is kept as a
	 * fallback for the rare video that has no adaptive rendition.
	 */
	override suspend fun streams(itemId: String): Result<PlaybackStreams> =
		withContext(Dispatchers.IO) {
			runCatching {
				val info = StreamInfo.getInfo(service, watchUrl(itemId))

				val video = info.videoOnlyStreams
					// AV1 and VP9 decode fine on modern hardware, but H.264 is the
					// safest bet across cheap devices and emulators.
					.filter { it.content.isNotBlank() }
					.maxByOrNull { it.height }
				val audio = info.audioStreams
					.filter { it.content.isNotBlank() }
					.maxByOrNull { it.averageBitrate }

				if (video != null && audio != null) {
					PlaybackStreams.Split(
						videoUrl = video.content,
						audioUrl = audio.content,
						height = video.height,
						label = "${video.height}p",
					)
				} else {
					val muxed = info.videoStreams
						.filter { it.content.isNotBlank() }
						.maxByOrNull { it.height }
						?: error("no playable stream for $itemId")
					PlaybackStreams.Single(muxed.content, label = "${muxed.height}p")
				}
			}
		}

	private fun StreamInfoItem.toFeedItem(): FeedItem? {
		val videoId = idFromUrl(url) ?: return null
		return FeedItem(
			id = videoId,
			source = id,
			url = url,
			title = name.orEmpty(),
			author = uploaderName.orEmpty(),
			thumbnailUrl = thumbnails.best(),
			durationSeconds = duration,
			viewCount = viewCount,
		)
	}

	private fun List<Image>.best(): String =
		maxByOrNull { it.height }?.url ?: firstOrNull()?.url.orEmpty()
}
