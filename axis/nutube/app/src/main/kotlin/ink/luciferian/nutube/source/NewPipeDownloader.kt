package ink.luciferian.nutube.source

import okhttp3.OkHttpClient
import okhttp3.Request as OkRequest
import okhttp3.RequestBody.Companion.toRequestBody
import org.schabi.newpipe.extractor.downloader.Downloader
import org.schabi.newpipe.extractor.downloader.Request
import org.schabi.newpipe.extractor.downloader.Response
import java.util.concurrent.TimeUnit

/**
 * NewPipeExtractor's network hook, on OkHttp.
 *
 * OkHttp brings connection pooling, HTTP/2, retries and IPv4/IPv6 racing for free -
 * the last of which is exactly what the Godot prototype hand-rolled around by
 * resolving IPv4 itself and overriding the TLS common name. That workaround would
 * have broken on IPv6-only carrier networks.
 */
object NewPipeDownloader : Downloader() {

	private const val USER_AGENT =
		"Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0"

	private val client = OkHttpClient.Builder()
		.connectTimeout(15, TimeUnit.SECONDS)
		.readTimeout(20, TimeUnit.SECONDS)
		.build()

	override fun execute(request: Request): Response {
		val builder = OkRequest.Builder()
			.url(request.url())
			.method(
				request.httpMethod(),
				request.dataToSend()?.toRequestBody(),
			)
			.header("User-Agent", USER_AGENT)

		request.headers().forEach { (name, values) ->
			builder.removeHeader(name)
			values.forEach { builder.addHeader(name, it) }
		}

		client.newCall(builder.build()).execute().use { response ->
			return Response(
				response.code,
				response.message,
				response.headers.toMultimap(),
				response.body.string(),
				response.request.url.toString(),
			)
		}
	}
}
