# NewPipeExtractor drives Rhino reflectively to run YouTube's player JS.
-keep class org.mozilla.javascript.** { *; }
-keep class org.schabi.newpipe.extractor.** { *; }
-dontwarn org.mozilla.javascript.**
-dontwarn org.schabi.newpipe.extractor.**
# Its models are deserialized by name.
-keepclassmembers class org.schabi.newpipe.extractor.** { <init>(...); }
