/**
 * Send an iMessage via Photon Advanced iMessage SDK.
 * Usage: node photon_send.mjs <address> <token> <to> <message>
 *   address  — Photon gRPC server address, e.g. "api.photon.ai:443"
 *   token    — Bearer token from Photon dashboard
 *   to       — Recipient phone number or Apple ID email, e.g. "+14155550123"
 *   message  — Text to send (wrap in quotes)
 */
import { createClient } from "@photon-ai/advanced-imessage";

const [, , address, token, to, ...messageParts] = process.argv;
const message = messageParts.join(" ");

if (!address || !token || !to || !message) {
  console.error(
    "Usage: node photon_send.mjs <address> <token> <to> <message>"
  );
  process.exit(1);
}

const im = createClient({ address, token });

try {
  const sent = await im.messages.sendText(to, message);
  console.log(JSON.stringify({ ok: true, guid: sent.guid }));
} catch (err) {
  console.error(JSON.stringify({ ok: false, error: err.message }));
  process.exit(1);
} finally {
  await im.close();
}
