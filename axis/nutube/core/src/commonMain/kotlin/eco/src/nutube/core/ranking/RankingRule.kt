package eco.src.nutube.core.ranking

import eco.src.nutube.core.FeedItem

/** Everything a rule is allowed to look at. */
data class RankingContext(
	/** Tokenised search box contents; empty when the user is just browsing. */
	val queryTerms: List<String>,
	val affinity: Affinity,
)

/**
 * One reason a video might be worth surfacing.
 *
 * Rules are deliberately small and separable: each returns a score in 0..1 and a
 * sentence explaining itself, and the [Ranker] combines them. Adding a signal
 * means adding a rule to the registry, not editing a scoring function - and
 * because every rule can explain itself, the feed never has to say "because the
 * algorithm said so".
 */
interface RankingRule {

	val id: String

	/** How much this rule counts relative to the others. */
	val weight: Double

	/** 0..1. Anything outside that range breaks the comparison between rules. */
	fun score(item: FeedItem, ctx: RankingContext): Double

	/** Shown on the card when this rule is the reason the item rose. */
	fun reason(item: FeedItem, ctx: RankingContext): String
}

/**
 * Overlap between the search box and the item's own words.
 *
 * Only meaningful while the user is searching; with an empty box it stands down
 * entirely so the affinity rules decide the order.
 */
object QueryOverlapRule : RankingRule {
	override val id = "query"
	override val weight = 3.0

	override fun score(item: FeedItem, ctx: RankingContext): Double {
		if (ctx.queryTerms.isEmpty()) return 0.0
		val haystack = Tokens.words(
			item.title + " " + item.author + " " + item.tags.joinToString(" ")
		).toSet()
		return ctx.queryTerms.count { it in haystack }.toDouble() / ctx.queryTerms.size
	}

	override fun reason(item: FeedItem, ctx: RankingContext) = "Matches your search"
}

/**
 * How often this channel has been watched.
 *
 * Scaled against the most-watched channel rather than an absolute count, so the
 * rule means the same thing on day one as it does after a thousand videos.
 */
object ChannelAffinityRule : RankingRule {
	override val id = "channel"
	override val weight = 2.0

	override fun score(item: FeedItem, ctx: RankingContext): Double {
		val seen = ctx.affinity.channels[item.author] ?: return 0.0
		val peak = ctx.affinity.channelPeak
		if (peak <= 0) return 0.0
		return (seen.toDouble() / peak).coerceIn(0.0, 1.0)
	}

	override fun reason(item: FeedItem, ctx: RankingContext): String {
		val seen = ctx.affinity.channels[item.author] ?: 0
		return "You have watched ${item.author} $seen time${if (seen == 1) "" else "s"}"
	}
}

/**
 * Subject matter, learned from the titles of things already watched.
 *
 * Averages over the title's own n-grams rather than summing, so a long title
 * cannot out-score a short one just by containing more words.
 */
object TitleTermRule : RankingRule {
	override val id = "terms"
	override val weight = 1.5

	override fun score(item: FeedItem, ctx: RankingContext): Double {
		val peak = ctx.affinity.termPeak
		if (peak <= 0) return 0.0
		val grams = Tokens.ngrams(item.title).toSet()
		if (grams.isEmpty()) return 0.0
		val total = grams.sumOf { (ctx.affinity.titleTerms[it] ?: 0).toDouble() / peak }
		return (total / grams.size).coerceIn(0.0, 1.0)
	}

	override fun reason(item: FeedItem, ctx: RankingContext): String {
		// One prior sighting is a coincidence, not a pattern worth claiming.
		val best = Tokens.ngrams(item.title).toSet()
			.maxByOrNull { ctx.affinity.titleTerms[it] ?: 0 }
			?.takeIf { (ctx.affinity.titleTerms[it] ?: 0) >= 2 }
		return if (best == null) "Related to what you watch" else "You keep watching \"$best\""
	}
}

/**
 * The rules in play, in the order they were added.
 *
 * This is the seam the whole recommender grows through: watch time, recency,
 * co-occurrence, an on-device embedding - each arrives as one more entry here.
 */
val RULE_REGISTRY: List<RankingRule> = listOf(
	QueryOverlapRule,
	ChannelAffinityRule,
	TitleTermRule,
)
