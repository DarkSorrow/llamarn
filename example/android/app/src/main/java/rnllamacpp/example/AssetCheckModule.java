package rnllamacpp.example;

import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.WritableArray;
import com.facebook.react.bridge.Arguments;

import java.io.IOException;
import java.io.InputStream;

public class AssetCheckModule extends ReactContextBaseJavaModule {

    public AssetCheckModule(ReactApplicationContext reactContext) {
        super(reactContext);
    }

    @Override
    public String getName() {
        return "AssetCheckModule";
    }

    @ReactMethod
    public void doesAssetExist(String fileName, Promise promise) {
        try {
            InputStream inputStream = getReactApplicationContext().getAssets().open(fileName);
            inputStream.close();
            promise.resolve(true);
        } catch (IOException e) {
            promise.resolve(false);
        } catch (Exception e) {
            promise.reject("ASSET_CHECK_ERROR", e.getMessage());
        }
    }

    @ReactMethod
    public void listAssets(String path, Promise promise) {
        try {
            String[] assets = getReactApplicationContext().getAssets().list(path);
            WritableArray assetArray = Arguments.createArray();
            for (String asset : assets) {
                assetArray.pushString(asset);
            }
            promise.resolve(assetArray);
        } catch (IOException e) {
            promise.reject("ASSET_LIST_ERROR", e.getMessage());
        } catch (Exception e) {
            promise.reject("ASSET_LIST_ERROR", e.getMessage());
        }
    }
} 