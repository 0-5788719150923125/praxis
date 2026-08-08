package eco.src.nutube

import android.app.Application
import eco.src.nutube.core.LocalIndex
import eco.src.nutube.core.Settings
import eco.src.nutube.core.SourceRegistry
import eco.src.nutube.core.TermBank
import eco.src.nutube.core.ranking.AffinityStore
import eco.src.nutube.sources.youtube.NewPipeDownloader
import eco.src.nutube.sources.youtube.YouTubeSource
import okio.Path.Companion.toOkioPath
import org.schabi.newpipe.extractor.NewPipe
import org.schabi.newpipe.extractor.localization.ContentCountry
import org.schabi.newpipe.extractor.localization.Localization

class NuTubeApp : Application() {

	lateinit var index: LocalIndex
		private set

	/** Saved searches - the subscription list the index is built from. */
	lateinit var terms: TermBank
		private set

	lateinit var settings: Settings
		private set

	/** What the device has learned from what you actually open. */
	lateinit var affinity: AffinityStore
		private set

	override fun onCreate() {
		super.onCreate()
		// NewPipeExtractor is a singleton and must be handed a Downloader before use.
		NewPipe.init(NewPipeDownloader, Localization("en", "US"), ContentCountry("US"))

		// Platforms plug in here. commonMain never names a concrete source.
		SourceRegistry.register(YouTubeSource)

		index = LocalIndex(filesDir.resolve("index.json").toOkioPath())
		terms = TermBank(filesDir.resolve("terms.json").toOkioPath())
		settings = Settings(filesDir.resolve("settings.json").toOkioPath())
		affinity = AffinityStore(filesDir.resolve("affinity.json").toOkioPath())
	}
}
