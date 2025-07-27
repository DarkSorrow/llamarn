#!/bin/bash

echo "🧪 Testing release build with ProGuard for @novastera-oss/llamarn..."

# Build release APK using React Native CLI (which handles bundling correctly)
echo "📦 Building release APK using React Native CLI..."
npx react-native run-android --mode=release

if [ $? -eq 0 ]; then
    echo "✅ Release build successful!"
    
    # Check if ProGuard processed the classes
    echo "🔍 Checking ProGuard output..."
    
    if [ -f "android/app/build/outputs/mapping/release/mapping.txt" ]; then
        echo "✅ ProGuard mapping file found"
        
        # Check if our library classes are preserved
        echo "🔍 Checking if library classes are preserved..."
        if grep -q "com.novastera.llamarn" android/app/build/outputs/mapping/release/mapping.txt; then
            echo "✅ Library classes found in mapping (preserved by ProGuard rules)"
            echo "📋 Sample of preserved classes:"
            grep "com.novastera.llamarn" android/app/build/outputs/mapping/release/mapping.txt | head -5
        else
            echo "⚠️  Library classes not found in mapping"
        fi
        
        # Check seeds file
        if [ -f "android/app/build/outputs/mapping/release/seeds.txt" ]; then
            echo "🌱 Checking seeds file..."
            if grep -q "com.novastera.llamarn" android/app/build/outputs/mapping/release/seeds.txt; then
                echo "✅ Library classes found in seeds (explicitly kept)"
            else
                echo "⚠️  Library classes not found in seeds"
            fi
        fi
        
        # Check usage file
        if [ -f "android/app/build/outputs/mapping/release/usage.txt" ]; then
            echo "🗑️  Checking usage file (removed classes)..."
            if grep -q "com.novastera.llamarn" android/app/build/outputs/mapping/release/usage.txt; then
                echo "❌ Library classes found in usage (removed by ProGuard)"
            else
                echo "✅ Library classes not found in usage (preserved)"
            fi
        fi
    else
        echo "❌ ProGuard mapping file not found"
    fi
    
    # Check APK size
    if [ -f "android/app/build/outputs/apk/release/app-release.apk" ]; then
        APK_SIZE=$(ls -lh android/app/build/outputs/apk/release/app-release.apk | awk '{print $5}')
        echo "📱 Release APK size: $APK_SIZE"
    fi
    
else
    echo "❌ Release build failed!"
    exit 1
fi

echo "🎉 Release build test completed!" 