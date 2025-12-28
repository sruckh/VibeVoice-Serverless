# Cloudflare Worker Deployment Guide

This guide walks you through deploying the OpenAI TTS API bridge as a Cloudflare Worker.

## Overview

The Cloudflare Worker acts as a translation layer between the OpenAI Text-to-Speech API format and the VibeVoice RunPod serverless endpoint. This allows you to use OpenAI-compatible clients and libraries with your VibeVoice deployment.

**Flow:**
```
OpenAI TTS Client → Cloudflare Worker → RunPod VibeVoice → Audio Output
```

## Prerequisites

1. **Cloudflare Account** - [Sign up for free](https://dash.cloudflare.com/sign-up)
2. **Wrangler CLI** - Cloudflare's command-line tool
3. **RunPod Endpoint** - Your deployed VibeVoice serverless endpoint
4. **Voice Files** - `.wav` files uploaded to RunPod volume

## Step 1: Install Wrangler CLI

```bash
# Install globally via npm
npm install -g wrangler

# Or use npx (no installation needed)
npx wrangler --version

# Login to Cloudflare
wrangler login
```

This will open a browser window to authenticate with your Cloudflare account.

## Step 2: Create R2 Bucket

The worker uses Cloudflare R2 (S3-compatible storage) to store voice mappings.

```bash
# Create R2 bucket (if you haven't already)
wrangler r2 bucket create vibevoice

# Verify bucket exists
wrangler r2 bucket list
```

## Step 3: Configure Voice Mappings

Create a `voices.json` file that maps OpenAI voice names to your VibeVoice speaker names:

**File: `voices.json`**
```json
{
  "alloy": "Dorota",
  "echo": "Carter",
  "fable": "Frank",
  "onyx": "Mary",
  "nova": "Maya",
  "shimmer": "Samuel"
}
```

**Important:** The values (right side) must match the `.wav` filenames in `/runpod-volume/vibevoice/demo/voices/` (without the `.wav` extension).

Upload to R2 (**IMPORTANT: use `--remote` flag**):
```bash
cd bridge/
wrangler r2 object put vibevoice/voices.json --file=voices.json --remote
```

Verify upload (this will display the JSON content):
```bash
wrangler r2 object get vibevoice/voices.json --remote --pipe
```

**Critical:** Always use the `--remote` flag! Without it, Wrangler uploads to a local development R2 instance, and your deployed worker won't be able to access the file.

If you see your voice mappings JSON, the upload was successful!

## Step 4: Configure Worker

Copy the example configuration:

```bash
cd bridge/
cp wrangler.toml.example wrangler.toml
```

Edit `wrangler.toml`:

```toml
name = "vibevoice-openai-bridge"
main = "worker.js"
compatibility_date = "2024-12-18"

[[r2_buckets]]
binding = "VIBEVOICE_BUCKET"
bucket_name = "vibevoice"  # ← Your R2 bucket name

[vars]
# ← Your RunPod endpoint URL (get from RunPod dashboard)
RUNPOD_URL = "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync"
```

**Get your RunPod URL:**
1. Go to [RunPod Serverless Dashboard](https://www.runpod.io/console/serverless)
2. Click on your VibeVoice endpoint
3. Copy the endpoint URL (format: `https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync`)

## Step 5: Set Secrets

Secrets should **never** be committed to git or stored in `wrangler.toml`. Use the Wrangler CLI to set them securely:

### Required Secret: RUNPOD_API_KEY

```bash
wrangler secret put RUNPOD_API_KEY
# Enter your RunPod API key when prompted
```

**Get your RunPod API key:**
1. Go to [RunPod Settings](https://www.runpod.io/console/user/settings)
2. Navigate to "API Keys"
3. Copy your API key or create a new one

### Optional Secret: AUTH_TOKEN

If you want to protect your worker with authentication:

```bash
wrangler secret put AUTH_TOKEN
# Enter a secure random token when prompted
# Example: Use `openssl rand -hex 32` to generate one
```

If `AUTH_TOKEN` is set, clients must include it in requests:
```bash
curl https://your-worker.workers.dev/v1/audio/speech \
  -H "Authorization: Bearer YOUR_AUTH_TOKEN" \
  ...
```

If `AUTH_TOKEN` is **not** set, the worker is publicly accessible (anyone can use it).

## Step 6: Deploy Worker

```bash
cd bridge/
wrangler deploy
```

**Expected output:**
```
✨ Build succeeded
✨ Successfully published your script to
   https://vibevoice-openai-bridge.YOUR_SUBDOMAIN.workers.dev
```

**Note:** Cloudflare will assign a subdomain automatically. You can also configure a custom domain in the Cloudflare dashboard.

## Step 7: Test the Deployment

### Test 1: Health Check

```bash
curl https://vibevoice-openai-bridge.YOUR_SUBDOMAIN.workers.dev/v1/audio/speech \
  -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_AUTH_TOKEN" \
  -d '{
    "model": "tts-1",
    "input": "Hello! This is a test of the OpenAI TTS bridge.",
    "voice": "alloy"
  }' \
  --output test.mp3
```

### Test 2: Verify Audio

```bash
# Check file size (should be > 0 bytes)
ls -lh test.mp3

# Play audio (macOS)
afplay test.mp3

# Play audio (Linux with mpg123)
mpg123 test.mp3

# Play audio (Windows with ffplay)
ffplay test.mp3
```

### Test 3: Try Different Voices

```bash
# Test with different OpenAI voice names
for voice in alloy echo fable onyx nova shimmer; do
  echo "Testing voice: $voice"
  curl https://your-worker.workers.dev/v1/audio/speech \
    -X POST \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer YOUR_AUTH_TOKEN" \
    -d "{\"model\":\"tts-1\",\"input\":\"Hello from $voice\",\"voice\":\"$voice\"}" \
    --output "test_$voice.mp3"
done
```

## Step 8: Use with OpenAI SDK

The worker is now compatible with OpenAI's official SDKs:

### Python Example

```python
from openai import OpenAI

# Point to your Cloudflare Worker
client = OpenAI(
    api_key="YOUR_AUTH_TOKEN",  # Your worker's AUTH_TOKEN
    base_url="https://your-worker.workers.dev"
)

response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="Hello! This is VibeVoice speaking through the OpenAI API."
)

response.stream_to_file("speech.mp3")
```

### JavaScript Example

```javascript
import OpenAI from "openai";

const openai = new OpenAI({
  apiKey: "YOUR_AUTH_TOKEN",
  baseURL: "https://your-worker.workers.dev"
});

const mp3 = await openai.audio.speech.create({
  model: "tts-1",
  voice: "alloy",
  input: "Hello! This is VibeVoice speaking through the OpenAI API."
});

const buffer = Buffer.from(await mp3.arrayBuffer());
await fs.promises.writeFile("speech.mp3", buffer);
```

## Management Commands

### View Logs

```bash
# Stream live logs
wrangler tail

# View logs in dashboard
# Go to: Workers & Pages → vibevoice-openai-bridge → Logs
```

### Update Secrets

```bash
# Update RunPod API key
wrangler secret put RUNPOD_API_KEY

# Update auth token
wrangler secret put AUTH_TOKEN

# List secrets (doesn't show values)
wrangler secret list
```

### Update Voice Mappings

```bash
# Edit voices.json locally
nano voices.json

# Upload to R2
wrangler r2 object put vibevoice/voices.json --file=voices.json

# Cache refreshes automatically within 5 minutes
```

### Redeploy

```bash
# After making changes to worker.js
wrangler deploy
```

### Delete Worker

```bash
wrangler delete vibevoice-openai-bridge
```

## Troubleshooting

### Issue 1: "voices.json not found in R2 bucket"

**Cause:** Voice mappings file not uploaded to R2.

**Solution:**
```bash
# Verify bucket exists
wrangler r2 bucket list

# Upload voices.json
wrangler r2 object put vibevoice/voices.json --file=voices.json

# Verify upload
wrangler r2 object get vibevoice/voices.json
```

### Issue 2: "Missing Authorization header"

**Cause:** `AUTH_TOKEN` is set but not included in request.

**Solution:**
- Include `Authorization: Bearer YOUR_AUTH_TOKEN` header in all requests
- Or remove `AUTH_TOKEN` secret to make worker public:
  ```bash
  wrangler secret delete AUTH_TOKEN
  ```

### Issue 3: "RunPod service error: 401"

**Cause:** Invalid `RUNPOD_API_KEY`.

**Solution:**
```bash
# Update the secret
wrangler secret put RUNPOD_API_KEY

# Verify RUNPOD_URL in wrangler.toml is correct
cat wrangler.toml | grep RUNPOD_URL
```

### Issue 4: "Invalid voice 'alloy'"

**Cause:** Voice not defined in `voices.json` or file not uploaded.

**Solution:**
1. Check `voices.json` includes the voice name
2. Re-upload to R2:
   ```bash
   wrangler r2 object put vibevoice/voices.json --file=voices.json
   ```
3. Wait 5 minutes for cache to refresh, or redeploy worker

### Issue 5: Worker returns base64 instead of S3 URL

**Cause:** S3 is not configured on RunPod endpoint.

**Solution:** This is normal! The worker will fetch and decode base64 automatically. Audio will still be returned as raw MP3 bytes (OpenAI-compatible).

## Advanced Configuration

### Custom Domain

1. Go to Cloudflare Dashboard → Workers & Pages → vibevoice-openai-bridge
2. Click "Triggers" tab
3. Add custom domain (e.g., `tts.yourdomain.com`)
4. Update your client code to use the custom domain

### Rate Limiting

Add rate limiting to prevent abuse:

```toml
# In wrangler.toml
[limits]
cpu_ms = 50  # Max CPU time per request
```

### Environment-Specific Deployments

```bash
# Deploy to staging
wrangler deploy --env staging

# Deploy to production
wrangler deploy --env production
```

Configure in `wrangler.toml`:
```toml
[env.staging]
vars = { RUNPOD_URL = "https://api.runpod.ai/v2/staging-endpoint/runsync" }

[env.production]
vars = { RUNPOD_URL = "https://api.runpod.ai/v2/prod-endpoint/runsync" }
```

## Security Best Practices

1. **Always set AUTH_TOKEN** for production deployments
2. **Use custom domain** with HTTPS (Cloudflare provides this automatically)
3. **Rotate secrets regularly** (RUNPOD_API_KEY, AUTH_TOKEN)
4. **Monitor logs** for unusual activity
5. **Limit CORS origins** if possible (modify `worker.js`)
6. **Keep wrangler.toml** out of git (already in .gitignore)

## Cost Estimation

Cloudflare Workers pricing (as of 2024):
- **Free tier:** 100,000 requests/day
- **Paid tier:** $5/month for 10M requests
- **R2 storage:** Free tier includes 10GB storage

For most use cases, the free tier is sufficient!

## Support

- **Cloudflare Workers Docs:** https://developers.cloudflare.com/workers/
- **Wrangler CLI Docs:** https://developers.cloudflare.com/workers/wrangler/
- **R2 Docs:** https://developers.cloudflare.com/r2/

---

**Questions or Issues?** Check the [main README](../README.md) or open a GitHub issue.
