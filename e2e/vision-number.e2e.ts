import { test } from '@e2e-dev/mobile';
import { expect } from 'e2e';

test('vision screen loads the number image', async ({ app, device, screen }) => {
  await device.installApp(undefined, { reinstall: true });
  await app.open();
  const allow = screen.getByRole('button', 'Allow');
  if (await allow.isVisible()) {
    await allow.tap();
  }
  await screen.getByRole('button', /vision chat demo/i).tap();
  await screen.getByTestId('vision-use-image').tap();
  await expect(screen.getByTestId('vision-status')).toContainText('Test image ready: 7');
  await expect(screen.getByTestId('vision-preview')).toBeVisible();
});
