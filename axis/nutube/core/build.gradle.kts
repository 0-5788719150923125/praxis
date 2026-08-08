plugins {
	alias(libs.plugins.kotlin.multiplatform)
	alias(libs.plugins.android.kmp.library)
	alias(libs.plugins.kotlin.serialization)
}

/**
 * The platform-free half of nuTube: the item model, the on-device index and
 * ranker, and the source registry every platform plugs into.
 *
 * Nothing here may reference an Android or JVM API. Adding a target below is the
 * whole change needed to run this logic somewhere else - what each new target
 * then owes is a `VideoSource` implementation and a player.
 */
kotlin {
	android {
		namespace = "eco.src.nutube.core"
		compileSdk = 37
		minSdk = 26
	}

	// iosArm64() / jvm() / wasmJs() go here when those targets land.

	sourceSets {
		commonMain.dependencies {
			implementation(libs.kotlinx.coroutines.core)
			implementation(libs.kotlinx.serialization.json)
			// Multiplatform filesystem, so LocalIndex never needs java.io.
			api(libs.okio)
		}
	}
}
