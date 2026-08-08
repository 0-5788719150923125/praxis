package eco.src.nutube.core

data class LegalLink(val label: String, val url: String)

/**
 * What a platform requires be said and shown about playing its videos.
 *
 * Each platform sets its own terms, so this travels with the [VideoSource]
 * rather than sitting in the app's settings screen. A future PeerTube source
 * has no advertising to speak of and different obligations entirely; the
 * settings screen just renders whatever each source declares.
 */
data class PlatformTerms(
	/** What using the platform's own embedded player means. */
	val embedNote: String,
	/** What bypassing it means, stated plainly. */
	val nativeNote: String,
	/**
	 * Links the platform requires be shown. YouTube's API Services Terms forbid
	 * removing or obscuring them, so nothing here may be collapsed or hidden.
	 */
	val links: List<LegalLink>,
)
