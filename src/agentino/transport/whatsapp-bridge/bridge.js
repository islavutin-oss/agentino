/**
 * Agentino WhatsApp Bridge
 *
 * Lightweight WhatsApp ↔ HTTP bridge using Baileys.
 * Forwards incoming messages to an agentino gateway and sends replies back.
 *
 * Usage:
 *   npm install && node bridge.js
 *
 * Environment:
 *   AGENTINO_URL   - URL of the agentino WhatsApp channel (default: http://localhost:8080)
 *   BRIDGE_PORT    - Port for this bridge's HTTP API (default: 3001)
 *   PAIRING_PHONE  - Phone number for pairing code auth (leave empty for QR code)
 *   AUTH_DIR       - Directory for WhatsApp auth state (default: ./auth)
 *   WHITELIST      - Comma-separated phone numbers to allow (empty = allow all)
 *   BLACKLIST      - Comma-separated phone numbers to block
 */

const { default: makeWASocket, useMultiFileAuthState, DisconnectReason, Browsers } = require('@whiskeysockets/baileys');
const qrcode = require('qrcode-terminal');
const QRCode = require('qrcode');
const pino = require('pino');
const express = require('express');

require('dotenv').config();

// Configuration
const AGENTINO_URL = process.env.AGENTINO_URL || 'http://localhost:8080';
const AUTH_DIR = process.env.AUTH_DIR || './auth';
const BRIDGE_PORT = process.env.BRIDGE_PORT || 3001;
const PAIRING_PHONE = process.env.PAIRING_PHONE || '';

// Global state
let globalSock = null;
let currentQRDataURL = null;
let currentPairingCode = null;
let reconnectDelay = 3000;
let skipPairingOnce = false;
let pendingRestart = null;

// Whitelist / blacklist
const normalizePhone = (phone) => String(phone || '').replace(/\D/g, '');
let whitelistNumbers = (process.env.WHITELIST || '').split(',').map(normalizePhone).filter(Boolean);
let blacklistNumbers = (process.env.BLACKLIST || '').split(',').map(normalizePhone).filter(Boolean);

function scheduleRestart(delay) {
    if (pendingRestart) clearTimeout(pendingRestart);
    pendingRestart = setTimeout(async () => {
        pendingRestart = null;
        await startBridge().catch(console.error);
    }, delay);
}

function phoneFromJid(jid) {
    if (!jid || typeof jid !== 'string' || !jid.endsWith('@s.whatsapp.net')) return null;
    const digits = normalizePhone(jid.split('@')[0].replace(/:.*/, ''));
    return /^\d{10,15}$/.test(digits) ? digits : null;
}

function findPhoneInObject(obj, depth = 0) {
    if (!obj || depth > 4) return null;
    if (typeof obj === 'string') return phoneFromJid(obj);
    if (Array.isArray(obj)) {
        for (const item of obj) {
            const found = findPhoneInObject(item, depth + 1);
            if (found) return found;
        }
        return null;
    }
    if (typeof obj === 'object') {
        for (const value of Object.values(obj)) {
            const found = findPhoneInObject(value, depth + 1);
            if (found) return found;
        }
    }
    return null;
}

console.log(`
  Agentino WhatsApp Bridge
  Agentino URL: ${AGENTINO_URL}
  Bridge port:  ${BRIDGE_PORT}
  Auth mode:    ${PAIRING_PHONE ? `Pairing code for ${PAIRING_PHONE}` : 'QR code'}
`);

// ============================================
// Forward message to agentino
// ============================================

async function forwardToAgentino(senderId, phone, message, senderName) {
    try {
        const response = await fetch(`${AGENTINO_URL}/message`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                sender_id: senderId,
                phone: phone,
                message: message,
                sender_name: senderName || null,
            }),
        });
        if (!response.ok) {
            console.error(`Agentino error: ${response.status}`);
            return null;
        }
        const data = await response.json();
        return data.response || null;
    } catch (error) {
        console.error('Agentino call failed:', error.message);
        return null;
    }
}

// ============================================
// HTTP API (outbound messages + status)
// ============================================

const app = express();
app.use(express.json());

app.post('/send', async (req, res) => {
    const { phone, message } = req.body;
    if (!phone || !message) return res.status(400).json({ error: 'Missing phone or message' });
    if (!globalSock) return res.status(503).json({ error: 'WhatsApp not connected' });
    try {
        const jid = `${phone.replace(/\D/g, '')}@s.whatsapp.net`;
        await globalSock.sendMessage(jid, { text: message });
        res.json({ success: true });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.get('/health', (req, res) => {
    res.json({ status: globalSock ? 'connected' : 'disconnected' });
});

app.get('/qr', (req, res) => {
    const meId = globalSock?.authState?.creds?.me?.id || null;
    const connectedPhone = meId ? '+' + meId.replace(/:.*/, '').replace('@s.whatsapp.net', '') : null;
    if (globalSock) return res.json({ status: 'connected', qr: null, pairing_code: null, connected_phone: connectedPhone });
    if (currentPairingCode) return res.json({ status: 'pairing_code', qr: null, pairing_code: currentPairingCode });
    if (currentQRDataURL) return res.json({ status: 'waiting_scan', qr: currentQRDataURL });
    return res.json({ status: 'initializing' });
});

app.post('/reconnect', async (req, res) => {
    try {
        if (globalSock) { globalSock.end(); globalSock = null; }
        currentQRDataURL = null;
        currentPairingCode = null;
        reconnectDelay = 3000;
        res.json({ success: true });
        scheduleRestart(1000);
    } catch (err) { res.status(500).json({ error: err.message }); }
});

app.post('/disconnect', async (req, res) => {
    const fs = require('fs');
    const path = require('path');
    try {
        if (globalSock) { await globalSock.logout().catch(() => {}); globalSock = null; }
        currentQRDataURL = null;
        currentPairingCode = null;
        reconnectDelay = 3000;
        const authDir = path.resolve(AUTH_DIR);
        if (fs.existsSync(authDir)) {
            for (const file of fs.readdirSync(authDir)) {
                fs.rmSync(path.join(authDir, file), { recursive: true, force: true });
            }
        }
        res.json({ success: true });
        scheduleRestart(1000);
    } catch (err) { res.status(500).json({ error: err.message }); }
});

app.listen(BRIDGE_PORT, () => {
    console.log(`  HTTP API: http://localhost:${BRIDGE_PORT}`);
    console.log(`    POST /send  { phone, message }`);
    console.log(`    GET  /health`);
    console.log(`    GET  /qr`);
    console.log('');
});

// ============================================
// WhatsApp Connection
// ============================================

async function startBridge() {
    const { state, saveCreds } = await useMultiFileAuthState(AUTH_DIR);

    const sock = makeWASocket({
        auth: state,
        logger: pino({ level: 'silent' }),
        version: [2, 3000, 1034074495],
        browser: Browsers.macOS('Chrome'),
        markOnlineOnConnect: false,
        syncFullHistory: false,
    });

    const usePairing = PAIRING_PHONE && !skipPairingOnce;
    skipPairingOnce = false;
    if (usePairing && !sock.authState.creds.registered) {
        const phoneDigits = PAIRING_PHONE.replace(/\D/g, '');
        if (!currentPairingCode) {
            setTimeout(async () => {
                try {
                    const code = await sock.requestPairingCode(phoneDigits);
                    currentPairingCode = code;
                    console.log(`  Pairing code: ${code}  (for +${phoneDigits})`);
                    console.log(`  Enter in WhatsApp > Linked Devices > Link with phone number\n`);
                } catch (err) {
                    console.error('Failed to get pairing code:', err.message);
                }
            }, 3000);
        }
    }

    sock.ev.on('connection.update', (update) => {
        const { connection, lastDisconnect, qr } = update;

        if (qr && !usePairing) {
            console.log('\n  Scan this QR code with WhatsApp:\n');
            qrcode.generate(qr, { small: true });
            QRCode.toDataURL(qr).then(url => { currentQRDataURL = url; }).catch(() => {});
        }

        if (connection === 'close') {
            const statusCode = lastDisconnect?.error?.output?.statusCode;
            if (statusCode === DisconnectReason.loggedOut) {
                console.log('  Logged out. Clearing session, restarting in QR mode...');
                currentQRDataURL = null;
                currentPairingCode = null;
                reconnectDelay = 3000;
                skipPairingOnce = true;
                const fs = require('fs');
                const path = require('path');
                const authDir = path.resolve(AUTH_DIR);
                if (fs.existsSync(authDir)) {
                    for (const file of fs.readdirSync(authDir)) {
                        try { fs.rmSync(path.join(authDir, file), { recursive: true, force: true }); } catch {}
                    }
                }
                scheduleRestart(reconnectDelay);
            } else {
                console.log(`  Connection closed (code=${statusCode}). Reconnecting in ${reconnectDelay/1000}s...`);
                scheduleRestart(reconnectDelay);
                reconnectDelay = Math.min(reconnectDelay * 2, 30000);
            }
        } else if (connection === 'open') {
            globalSock = sock;
            currentQRDataURL = null;
            currentPairingCode = null;
            reconnectDelay = 3000;
            console.log('  Connected to WhatsApp!\n');
        }
    });

    sock.ev.on('creds.update', saveCreds);

    sock.ev.on('messages.upsert', async ({ messages, type }) => {
        if (type !== 'notify') return;

        for (const msg of messages) {
            if (!msg.message || msg.key.fromMe) continue;

            const isDirectMessage = msg.key.remoteJid.endsWith('@s.whatsapp.net') ||
                                    msg.key.remoteJid.endsWith('@lid');
            if (!isDirectMessage) continue;

            const senderId = msg.key.remoteJid;
            const isLid = msg.key.remoteJid.endsWith('@lid');
            let phone = null;
            if (!isLid) {
                phone = phoneFromJid(msg.key.remoteJid);
            } else {
                phone = phoneFromJid(msg.key.participant) ||
                        phoneFromJid(msg.participant) ||
                        findPhoneInObject(msg.message);
            }

            const senderName = msg.pushName || null;
            const text = msg.message.conversation ||
                        msg.message.extendedTextMessage?.text || '';
            if (!text) continue;

            const senderAlias = phone || normalizePhone(senderId);
            if (blacklistNumbers.length > 0 && blacklistNumbers.includes(senderAlias)) continue;
            if (whitelistNumbers.length > 0 && !whitelistNumbers.includes(senderAlias)) continue;

            console.log(`  <- ${senderName || senderAlias}: ${text}`);

            // Forward to agentino — agentino handles reply via bridge /send
            const response = await forwardToAgentino(senderId, phone, text, senderName);

            // If agentino didn't send via /send (e.g. response returned inline), send here
            if (response) {
                await sock.sendMessage(msg.key.remoteJid, { text: response });
                console.log(`  -> ${response.substring(0, 60)}...`);
            }
            console.log('');
        }
    });
}

startBridge().catch(console.error);
