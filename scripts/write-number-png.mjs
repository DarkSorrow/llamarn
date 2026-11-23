#!/usr/bin/env node
// 64x64 RGB PNG of the digit 7. No image library: one uncompressed-filter scanline stream, zlib, and a PNG CRC.
import { deflateSync } from 'node:zlib';
import { writeFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const SIZE = 64;
const SCALE = 6;
const GLYPH = [
  '11111',
  '00001',
  '00010',
  '00100',
  '01000',
  '10000',
  '10000',
];
const ORIGIN_X = Math.floor((SIZE - GLYPH[0].length * SCALE) / 2);
const ORIGIN_Y = Math.floor((SIZE - GLYPH.length * SCALE) / 2);

function pixel(x, y) {
  const gx = Math.floor((x - ORIGIN_X) / SCALE);
  const gy = Math.floor((y - ORIGIN_Y) / SCALE);
  if (gy < 0 || gy >= GLYPH.length || gx < 0 || gx >= GLYPH[0].length) {
    return 255;
  }
  return GLYPH[gy][gx] === '1' ? 0 : 255;
}

const crcTable = new Uint32Array(256);
for (let n = 0; n < 256; n++) {
  let c = n;
  for (let k = 0; k < 8; k++) {
    c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
  }
  crcTable[n] = c >>> 0;
}

function crc32(buf) {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) {
    c = crcTable[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  }
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(type, data) {
  const body = Buffer.concat([Buffer.from(type), data]);
  const out = Buffer.alloc(12 + data.length);
  out.writeUInt32BE(data.length, 0);
  body.copy(out, 4);
  out.writeUInt32BE(crc32(body), 8 + data.length);
  return out;
}

function png() {
  const raw = Buffer.alloc(SIZE * (1 + SIZE * 3));
  for (let y = 0; y < SIZE; y++) {
    const row = y * (1 + SIZE * 3);
    raw[row] = 0;
    for (let x = 0; x < SIZE; x++) {
      const v = pixel(x, y);
      const i = row + 1 + x * 3;
      raw[i] = v;
      raw[i + 1] = v;
      raw[i + 2] = v;
    }
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(SIZE, 0);
  ihdr.writeUInt32BE(SIZE, 4);
  ihdr[8] = 8;
  ihdr[9] = 2;
  return Buffer.concat([
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
    chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(raw)),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

const topBarY = ORIGIN_Y;
const topBarX = ORIGIN_X + SCALE;
if (pixel(topBarX, topBarY) !== 0) {
  console.error('digit 7 top bar is not black');
  process.exit(1);
}

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const out = join(root, 'example/assets/number-7.png');
const androidAssets = join(root, 'example/android/app/src/main/assets');
const bytes = png();
writeFileSync(out, bytes);
mkdirSync(androidAssets, { recursive: true });
writeFileSync(join(androidAssets, 'number-7.png'), bytes);
console.log(out);
