import RNLlamaCpp from './NativeRNLlamaCpp';

export function multiply(a: number, b: number): number {
  return RNLlamaCpp.multiply(a, b);
}
