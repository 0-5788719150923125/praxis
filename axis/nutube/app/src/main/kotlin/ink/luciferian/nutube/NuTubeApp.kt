package ink.luciferian.nutube

import android.app.Application
import ink.luciferian.nutube.data.LocalIndex
import ink.luciferian.nutube.source.NewPipeDownloader
import org.schabi.newpipe.extractor.NewPipe
import org.schabi.newpipe.extractor.localization.ContentCountry
import org.schabi.newpipe.extractor.localization.Localization

class NuTubeApp : Application() {

	lateinit var index: LocalIndex
		private set

	override fun onCreate() {
		super.onCreate()
		// NewPipeExtractor is a singleton and must be handed a Downloader before use.
		NewPipe.init(NewPipeDownloader, Localization("en", "US"), ContentCountry("US"))
		index = LocalIndex(filesDir.resolve("index.json"))
	}
}
