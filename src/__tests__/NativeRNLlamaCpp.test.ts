jest.mock('react-native', () => ({
  TurboModuleRegistry: {
    getEnforcing: jest.fn(() => ({
      onModelLoadProgress: jest.fn(),
    })),
  },
}));

import { TurboModuleRegistry } from 'react-native';
import { addModelLoadProgressListener } from '../NativeRNLlamaCpp';

// `getEnforcing` is called exactly once, at NativeRNLlamaCpp's module-load
// time, to build the `LlamaCppRn` singleton. Grab the mock native module it
// returned so we can assert against the same `onModelLoadProgress` mock fn
// that `addModelLoadProgressListener` forwards into.
const getEnforcingMock = TurboModuleRegistry.getEnforcing as jest.Mock;
const nativeModule = getEnforcingMock.mock.results[0]!.value as {
  onModelLoadProgress: jest.Mock;
};

describe('addModelLoadProgressListener', () => {
  beforeEach(() => {
    nativeModule.onModelLoadProgress.mockReset();
  });

  it('forwards the listener to the native onModelLoadProgress emitter', () => {
    const listener = jest.fn();

    addModelLoadProgressListener(listener);

    expect(nativeModule.onModelLoadProgress).toHaveBeenCalledTimes(1);
    expect(nativeModule.onModelLoadProgress).toHaveBeenCalledWith(listener);
  });

  it('returns the EventSubscription produced by the native emitter', () => {
    const remove = jest.fn();
    nativeModule.onModelLoadProgress.mockReturnValue({ remove });

    const subscription = addModelLoadProgressListener(jest.fn());

    expect(subscription.remove).toBe(remove);
  });
});
