// Physical-device targets are a later opt-in. Do not add a paired iPhone or a USB
// Android phone to these two targets. Each `device` is a virtual device name.
import type { E2EConfig } from 'e2e';
import { mobile } from '@e2e-dev/mobile';

const metro = {
  executable: 'npm',
  args: ['--prefix', 'example', 'start'],
  reuseExisting: true,
};

const appServer = {
  command: metro,
  readyUrl: 'http://localhost:8081/status',
};

const iosApp = 'example/ios/build/Build/Products/Debug-iphonesimulator/llamarnexample.app';
const androidApp = 'example/android/app/build/outputs/apk/debug/app-debug.apk';

export default {
  workers: 1,
  timeout: 600_000,
  actionTimeout: 60_000,
  tests: 'e2e/**/*.e2e.ts',
  targets: [
    {
      name: 'ios',
      engine: mobile({ platform: 'ios', device: 'iPhone 17' }),
      app: {
        bundleId: 'org.reactjs.native.example.llamarnexample',
        appPath: iosApp,
        ...appServer,
      },
    },
    {
      name: 'android',
      engine: mobile({ platform: 'android', device: 'Pixel_10_Pro' }),
      app: {
        bundleId: 'rnllamacpp.example',
        appPath: androidApp,
        ...appServer,
      },
    },
  ],
} satisfies E2EConfig;
