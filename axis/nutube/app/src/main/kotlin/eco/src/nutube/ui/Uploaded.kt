package eco.src.nutube.ui

import android.text.format.DateUtils
import eco.src.nutube.core.FeedItem

/**
 * When a video went up, in words.
 *
 * Prefers a real timestamp so the phrasing is localised and stays correct as
 * time passes; falls back to whatever the platform said, since some listings
 * only ever give "2 years ago" and never a date.
 */
fun uploadedLabel(item: FeedItem): String = when {
	item.uploadedAt > 0 -> DateUtils.getRelativeTimeSpanString(
		item.uploadedAt,
		System.currentTimeMillis(),
		DateUtils.DAY_IN_MILLIS,
	).toString()

	else -> item.uploadedText
}
