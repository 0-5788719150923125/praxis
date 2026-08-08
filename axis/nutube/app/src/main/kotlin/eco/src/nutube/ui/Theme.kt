package eco.src.nutube.ui

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

// Carried over from the Godot prototype, which drew muted text at
// Color(0.6, 0.65, 0.72) over a near-black panel.
val Ink = Color(0xFF0F1115)
val Surface = Color(0xFF171A21)
val SurfaceHigh = Color(0xFF1F232C)
val Accent = Color(0xFFE8503A)
val Muted = Color(0xFF99A6B8)
val Bright = Color(0xFFE8EAF0)

private val scheme = darkColorScheme(
	primary = Accent,
	onPrimary = Ink,
	background = Ink,
	onBackground = Bright,
	surface = Surface,
	onSurface = Bright,
	surfaceVariant = SurfaceHigh,
	onSurfaceVariant = Muted,
	outline = Color(0xFF2A3039),
)

private val typography = Typography(
	titleMedium = TextStyle(fontSize = 17.sp, lineHeight = 23.sp, fontWeight = FontWeight.SemiBold),
	titleLarge = TextStyle(fontSize = 20.sp, fontWeight = FontWeight.Bold),
	bodySmall = TextStyle(fontSize = 13.sp),
	labelSmall = TextStyle(fontSize = 11.sp),
)

@Composable
fun NuTubeTheme(content: @Composable () -> Unit) =
	MaterialTheme(colorScheme = scheme, typography = typography, content = content)
