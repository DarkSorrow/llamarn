# ProGuard rules for @novastera-oss/llamarn library
# These rules will be automatically included when apps use this library

# Keep all classes in our package (includes NativeRNLlamaCppSpec, RNLlamaCppPackage, etc.)
-keep class com.novastera.llamarn.** {
    *;
}

# Keep native methods (JNI)
-keepclassmembers class com.novastera.llamarn.** {
    native <methods>;
}
