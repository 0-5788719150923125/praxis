package eco.src.nutube.core.ranking

/** Shared text handling, so the ranker and the affinity store always agree. */
object Tokens {

	private val SPLIT = Regex("[^\\p{L}\\p{N}]+")

	/**
	 * Words that say nothing about subject matter.
	 *
	 * Without this the title rule latches onto whatever is most common in English
	 * rather than what the person watches, and every card ends up explaining
	 * itself with "you watch a lot about the".
	 */
	private val STOPWORDS = setOf(
		"the", "and", "for", "you", "your", "with", "that", "this", "from", "are",
		"was", "were", "but", "not", "all", "can", "has", "have", "how", "what",
		"why", "when", "who", "will", "out", "our", "his", "her", "its", "they",
		"them", "their", "then", "than", "there", "here", "into", "onto", "over",
		"about", "just", "like", "get", "got", "one", "two", "new", "now", "very",
		"more", "most", "some", "any", "off", "official", "video", "full", "part",
	)

	/** Words worth counting: lowercased, punctuation dropped, singles discarded. */
	fun words(text: String): List<String> =
		text.lowercase().split(SPLIT).filter { it.length > 1 }

	/** [words] minus the ones that carry no subject matter. */
	fun contentWords(text: String): List<String> =
		words(text).filterNot { it in STOPWORDS }

	/**
	 * Unigrams plus bigrams, over content words only.
	 *
	 * Bigrams are what carry the actual subject - "lock" and "picking" separately
	 * say much less than "lock picking" - so a title contributes both and the
	 * ranker can reward whichever it has seen. Stopwords are dropped before
	 * pairing, so "history of rome" yields "history rome" rather than two pairs
	 * hinged on a word that means nothing.
	 */
	fun ngrams(text: String): List<String> {
		val w = contentWords(text)
		if (w.size < 2) return w
		return w + w.zipWithNext { a, b -> "$a $b" }
	}
}
