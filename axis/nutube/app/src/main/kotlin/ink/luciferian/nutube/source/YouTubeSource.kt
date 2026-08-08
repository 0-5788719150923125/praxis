package ink.luciferian.nutube.source

import ink.luciferian.nutube.data.FeedItem
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.schabi.newpipe.extractor.Image
import org.schabi.newpipe.extractor.ServiceList
import org.schabi.newpipe.extractor.search.SearchInfo
import org.schabi.newpipe.extractor.stream.StreamInfo
import org.schabi.newpipe.extractor.stream.StreamInfoItem

/**
 * YouTube behind the generic [FeedItem] shape, with no API key and no OAuth.
 *
 * NewPipeExtractor speaks YouTube's InnerTube protocol and runs the player's
 * JavaScript through Rhino to decipher stream URLs. That JS step is why this could
 * not have stayed in GDScript: there is no JS engine to run it in.
 */
object YouTubeSource {

	private val service = ServiceList.YouTube

	/** Extract the 11-char video id from the common YouTube URL shapes. */
	fun idFromUrl(url: String): String? =
		Regex("(?:v=|youtu\\.be/|/shorts/|/embed/)([A-Za-z0-9_-]{11})").find(url)?.groupValues?.get(1)

	fun watchUrl(id: String) = "https://www.youtube.com/watch?v=$id"

	suspend fun search(query: String, limit: Int = 20): Result<List<FeedItem>> =
		withContext(Dispatchers.IO) {
			runCatching {
				val handler = service.searchQHFactory.fromQuery(query, listOf("videos"), "")
				SearchInfo.getInfo(service, handler)
					.relatedItems
					.filterIsInstance<StreamInfoItem>()
					.take(limit)
					.map { it.toFeedItem() }
			}
		}

	suspend fun resolve(url: String): Result<FeedItem> = withContext(Dispatchers.IO) {
		runCatching {
			val id = idFromUrl(url) ?: error("not a YouTube video URL: $url")
			StreamInfo.getInfo(service, watchUrl(id)).let { info ->
				FeedItem(
					id = id,
					source = "youtube",
					url = watchUrl(id),
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
	 * A directly playable URL for [id].
	 *
	 * Prefers YouTube's HLS/DASH manifest so ExoPlayer can adapt bitrate; falls back
	 * to the best progressive muxed stream, which is capped at 720p by YouTube.
	 */
	suspend fun playbackUrl(id: String): Result<String> = withContext(Dispatchers.IO) {
		runCatching {
			val info = StreamInfo.getInfo(service, watchUrl(id))
			info.hlsUrl?.takeIf { it.isNotBlank() }
				?: info.dashMpdUrl?.takeIf { it.isNotBlank() }
				?: info.videoStreams.maxByOrNull { it.height }?.content
				?: error("no playable stream for $id")
		}
	}

	private fun StreamInfoItem.toFeedItem() = FeedItem(
		id = idFromUrl(url).orEmpty(),
		source = "youtube",
		url = url,
		title = name.orEmpty(),
		author = uploaderName.orEmpty(),
		thumbnailUrl = thumbnails.best(),
		durationSeconds = duration,
		viewCount = viewCount,
	)

	private fun List<Image>.best(): String =
		maxByOrNull { it.height }?.url ?: firstOrNull()?.url.orEmpty()
}
