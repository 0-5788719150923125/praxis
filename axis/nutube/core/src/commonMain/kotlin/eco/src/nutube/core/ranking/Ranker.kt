package eco.src.nutube.core.ranking

import eco.src.nutube.core.FeedItem

/**
 * Combines the rules into an order, and keeps the reason attached.
 *
 * The score is a weighted sum, but the *explanation* comes from whichever single
 * rule contributed most. That keeps the card honest - it names the actual reason
 * this video beat the others, rather than a summary of the arithmetic.
 */
class Ranker(private val rules: List<RankingRule> = RULE_REGISTRY) {

	fun rank(items: List<FeedItem>, ctx: RankingContext, limit: Int): List<FeedItem> {
		if (items.isEmpty()) return emptyList()

		val scored = items.map { item ->
			var total = 0.0
			var bestRule: RankingRule? = null
			var bestContribution = 0.0

			for (rule in rules) {
				val contribution = rule.score(item, ctx) * rule.weight
				total += contribution
				if (contribution > bestContribution) {
					bestContribution = contribution
					bestRule = rule
				}
			}
			Triple(item, total, bestRule)
		}

		return scored
			.sortedByDescending { it.second }
			.take(limit)
			.map { (item, _, rule) ->
				item.copy(
					reason = rule?.reason(item, ctx)
					// Nothing scored: the index is all there is to go on.
						?: if (ctx.queryTerms.isEmpty()) "From your local index"
						else "Loosely related"
				)
			}
	}
}
