package eco.src.nutube.core

/**
 * How a platform's videos are played.
 *
 * [EMBED] is the default everywhere. It uses the platform's own sanctioned
 * embedded player, which means the platform serves its own ads, counts the view,
 * and credits the creator - the behaviour a platform would approve of.
 *
 * [NATIVE] plays extracted streams directly. It is better software - hardware
 * decode, real quality control, background audio, picture-in-picture - and it
 * does none of those things for the creator. That trade is the user's to make,
 * knowingly, which is why it lives behind a setting and is never the default.
 */
enum class PlaybackMode { EMBED, NATIVE }
