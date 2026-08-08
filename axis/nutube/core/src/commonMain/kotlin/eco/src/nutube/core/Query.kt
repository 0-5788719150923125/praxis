package eco.src.nutube.core

/**
 * A search, after the prefixes have been read off it.
 *
 * `channel: @The-Arc` asks a platform for that channel's uploads instead of
 * running a keyword search. The prefix form exists so a follow is just a saved
 * term like any other - the Terms tab stays one list, the crawler stays one
 * loop, and following a channel needs no separate storage or UI.
 */
data class ParsedQuery(
	/** Channel identifier to enumerate, or null for an ordinary keyword search. */
	val channel: String?,
	/** Whatever was left after the prefix. */
	val text: String,
) {
	val isChannel: Boolean get() = channel != null
}

object Query {

	const val CHANNEL_PREFIX = "channel:"

	private val CHANNEL_ID = Regex("^UC[A-Za-z0-9_-]{22}$")

	fun parse(raw: String): ParsedQuery {
		val trimmed = raw.trim()

		// A leading @ is the ordinary way to name a YouTube channel, so it is the
		// prefix: `@veritasium` reads better than `channel: veritasium` and is
		// what a person already knows. Only a leading @ counts, so an address or
		// a handle mentioned mid-sentence stays an ordinary search.
		if (trimmed.startsWith("@") && trimmed.length > 1) {
			return ParsedQuery(channel = trimmed, text = trimmed)
		}

		// `channel:` stays as an alias, because a channel URL or a UC... id cannot
		// be written with an @ and following from a video has only the URL.
		if (!trimmed.startsWith(CHANNEL_PREFIX, ignoreCase = true)) {
			return ParsedQuery(channel = null, text = trimmed)
		}
		val name = trimmed.substring(CHANNEL_PREFIX.length).trim().trim('"')
		return if (name.isEmpty()) ParsedQuery(null, trimmed)
		else ParsedQuery(channel = name, text = name)
	}

	/**
	 * True when [identifier] names exactly one channel rather than describing one.
	 *
	 * A display name does not: several channels can be called "Arc", and a name
	 * search returns whichever the platform ranks first. A handle, a channel id
	 * and a channel URL each identify one channel and nothing else.
	 */
	fun isExactChannel(identifier: String): Boolean {
		val id = identifier.trim()
		return id.startsWith("@") || id.startsWith("http") || CHANNEL_ID.matches(id)
	}

	/**
	 * The canonical term for following a channel.
	 *
	 * Prefers the handle, then the channel URL, and only falls back to the display
	 * name when the platform gave nothing better - because a name is ambiguous and
	 * a follow that resolves to somebody else's channel is worse than no follow.
	 */
	fun channelTerm(displayName: String, channelUrl: String = ""): String {
		val handle = handleFrom(channelUrl)
		if (handle != null) return handle
		// No handle to be had - the platform reports channels by id - so keep the
		// URL, which is at least unambiguous. The Terms tab shows the channel's
		// name over it, and guessing a handle from a display name would reinvent
		// exactly the fuzzy matching this is meant to avoid.
		val identifier = channelUrl.ifBlank { displayName.trim() }
		return if (identifier.startsWith("@")) identifier else "$CHANNEL_PREFIX $identifier"
	}

	/** `https://www.youtube.com/@The-Arc` to `@The-Arc`, or null if there is none. */
	fun handleFrom(channelUrl: String): String? {
		val at = channelUrl.indexOf("/@")
		if (at < 0) return null
		return "@" + channelUrl.substring(at + 2).substringBefore('/').substringBefore('?')
	}

	/** The identifier a channel term carries, or null if the term is a plain search. */
	fun channelOf(term: String): String? = parse(term).channel

	/**
	 * One spelling per channel.
	 *
	 * `channel:@The-Arc` typed by hand and `channel: @The-Arc` written by the
	 * follow button name the same channel, and without collapsing them the Terms
	 * tab grows a duplicate that follows the same videos twice.
	 */
	fun canonical(term: String): String {
		val channel = channelOf(term)?.lowercase() ?: return term.trim().lowercase()
		// The result has to parse back as a channel. A handle says so by itself; a
		// URL or a bare name does not, so it keeps the prefix. Dropping it left
		// URL-form follows looking like ordinary keyword searches - unremovable,
		// unfollowable, and re-run as text.
		return if (channel.startsWith("@")) channel else CHANNEL_PREFIX + channel
	}
}
