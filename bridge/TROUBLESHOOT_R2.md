# Troubleshooting R2 Dashboard Visibility

## The File is There (Verified)

If `wrangler r2 object get vibevoice/voices.json` works, your file **is uploaded correctly**. The dashboard visibility is just a UI issue.

## Verification Commands

**CRITICAL: Always use the `--remote` flag for production R2!**

```bash
# Get the file content from REMOTE R2 (proves it exists in production)
wrangler r2 object get vibevoice/voices.json --remote --pipe

# This should display your JSON content
# If you see your voice mappings, the file is in production R2!
```

## Common Issue: Local vs Remote R2

Wrangler has **two R2 instances**:
- **Local** (development) - Used by default
- **Remote** (production) - Requires `--remote` flag

Your deployed worker accesses **remote R2**, so you must upload with `--remote`:

```bash
# ✓ Correct - uploads to production
wrangler r2 object put vibevoice/voices.json --file=voices.json --remote

# ✗ Wrong - uploads to local dev instance only
wrangler r2 object put vibevoice/voices.json --file=voices.json
```

**Available R2 Commands:**
- `wrangler r2 object get <objectPath> --remote` - Fetch an object
- `wrangler r2 object put <objectPath> --file=<file> --remote` - Upload an object
- `wrangler r2 object delete <objectPath> --remote` - Delete an object

## Dashboard Viewing Steps

1. **Go to Cloudflare Dashboard** → https://dash.cloudflare.com
2. Click **R2** in the left sidebar
3. Click on the **"vibevoice"** bucket
4. You should see **"voices.json"** in the object list

### Common Dashboard Issues

**Issue 1: Not seeing any files**
- Solution: Click the refresh button (↻) in the R2 dashboard
- Try a hard refresh: `Cmd+Shift+R` (Mac) or `Ctrl+Shift+R` (Windows/Linux)

**Issue 2: Wrong view/filter**
- Make sure you're not in a filtered view
- Check the "Path prefix" field is empty

**Issue 3: Browser cache**
- Try incognito/private browsing mode
- Clear browser cache
- Try a different browser

**Issue 4: Dashboard lag**
- Wait 1-2 minutes and refresh
- The dashboard can take time to sync

## The Important Part: Worker Will Work

**The dashboard view doesn't matter!** What matters is:

✅ `wrangler r2 object get vibevoice/voices.json` works
✅ Your worker's R2 binding is configured
✅ The file exists and is readable

Your Cloudflare Worker will have **no problem** accessing the file, even if the dashboard doesn't show it perfectly.

## Test Worker Access

The real test is whether your **deployed worker** can read the file. Deploy your worker and check the logs:

```bash
# Deploy worker
wrangler deploy

# Watch logs
wrangler tail

# In another terminal, make a test request
curl https://your-worker.workers.dev/v1/audio/speech \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"test","voice":"alloy"}'
```

In the logs, you should see:
```
Loaded voice mappings: alloy, echo, fable, onyx, nova, shimmer
```

If you see that, **your R2 setup is perfect!** The dashboard is just being finicky.

## Alternative: Upload via Dashboard

If you really want to see it in the dashboard, try uploading via the UI:

1. Go to **R2** → **vibevoice** bucket
2. Click **"Upload"** button
3. Select your local `voices.json` file
4. Upload
5. It should appear immediately

**Note:** This might create a duplicate if the CLI upload is also there. You can delete the old one if needed.

## Re-upload if Needed

If you want to start fresh:

```bash
# Delete existing file
wrangler r2 object delete vibevoice/voices.json

# Re-upload
wrangler r2 object put vibevoice/voices.json --file=voices.json

# Verify
wrangler r2 object list vibevoice
```

## Bottom Line

**Don't worry about the dashboard!** As long as:
- ✅ `wrangler r2 object get vibevoice/voices.json` returns your JSON
- ✅ Your worker deploys without errors
- ✅ Worker logs show "Loaded voice mappings"

You're **100% good to go!** The Cloudflare R2 dashboard is known for having UI quirks, but the API (which your worker uses) works perfectly.
