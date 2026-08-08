plugins {
	alias(libs.plugins.android.application)
	alias(libs.plugins.kotlin.compose)
	alias(libs.plugins.kotlin.serialization)
}

android {
	namespace = "ink.luciferian.nutube"
	compileSdk = 37

	defaultConfig {
		applicationId = "ink.luciferian.nutube"
		minSdk = 26
		targetSdk = 37
		versionCode = 1
		versionName = "0.1"
	}

	buildTypes {
		release {
			isMinifyEnabled = true
			isShrinkResources = true
			proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
		}
	}

	compileOptions {
		// NewPipeExtractor uses java.time / java.util.stream, so desugar them for minSdk 26.
		isCoreLibraryDesugaringEnabled = true
		sourceCompatibility = JavaVersion.VERSION_17
		targetCompatibility = JavaVersion.VERSION_17
	}

	kotlin {
		compilerOptions {
			jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
		}
	}

	buildFeatures {
		compose = true
	}

	packaging {
		resources.excludes += setOf("/META-INF/{AL2.0,LGPL2.1}", "META-INF/DEPENDENCIES")
	}
}

dependencies {
	implementation(libs.androidx.core.ktx)
	implementation(libs.androidx.lifecycle.runtime.ktx)
	implementation(libs.androidx.lifecycle.viewmodel.compose)
	implementation(libs.androidx.lifecycle.runtime.compose)
	implementation(libs.androidx.activity.compose)

	implementation(platform(libs.androidx.compose.bom))
	implementation(libs.androidx.compose.ui)
	implementation(libs.androidx.compose.ui.graphics)
	implementation(libs.androidx.compose.ui.tooling.preview)
	implementation(libs.androidx.compose.material3)
	implementation(libs.androidx.compose.material.icons)
	debugImplementation(libs.androidx.compose.ui.tooling)

	implementation(libs.androidx.media3.exoplayer)
	implementation(libs.androidx.media3.exoplayer.dash)
	implementation(libs.androidx.media3.exoplayer.hls)
	implementation(libs.androidx.media3.ui)
	implementation(libs.androidx.media3.session)

	implementation(libs.androidx.work.runtime.ktx)

	implementation(libs.coil.compose)
	implementation(libs.coil.network.okhttp)
	implementation(libs.okhttp)

	implementation(libs.newpipe.extractor)
	implementation(libs.kotlinx.serialization.json)

	coreLibraryDesugaring(libs.desugar.jdk.libs)
}
