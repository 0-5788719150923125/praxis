pluginManagement {
	repositories {
		google()
		mavenCentral()
		gradlePluginPortal()
	}
}

dependencyResolutionManagement {
	repositoriesMode = RepositoriesMode.FAIL_ON_PROJECT_REPOS
	repositories {
		google()
		mavenCentral()
		// NewPipeExtractor is only published here.
		maven("https://jitpack.io")
	}
}

rootProject.name = "nuTube"
include(":core")
include(":app")
