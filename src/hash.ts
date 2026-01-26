import crypto from 'crypto';
export function sha256Base64(input: string): string {
  return crypto.createHash('sha256').update(input, 'utf8').digest('base64');
}