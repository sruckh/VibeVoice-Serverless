#!/bin/bash
# Troubleshooting script for Cloudflare Worker R2 connection

echo "=== Cloudflare Worker R2 Troubleshooting ==="
echo ""

echo "Step 1: Verify file exists in R2"
echo "Running: wrangler r2 object get vibevoice/voices.json"
wrangler r2 object get vibevoice/voices.json
if [ $? -eq 0 ]; then
    echo "✓ File exists and is readable via CLI"
else
    echo "✗ File NOT found via CLI - need to re-upload"
    exit 1
fi

echo ""
echo "Step 2: Check wrangler.toml configuration"
echo "Bucket binding:"
grep -A 2 "r2_buckets" wrangler.toml

echo ""
echo "Step 3: Verify bucket exists"
echo "Running: wrangler r2 bucket list"
wrangler r2 bucket list

echo ""
echo "Step 4: Check worker deployment status"
echo "Running: wrangler deployments list"
wrangler deployments list 2>&1 || echo "(Command may not be available in your wrangler version)"

echo ""
echo "Step 5: View recent worker logs"
echo "Run in a separate terminal: wrangler tail"
echo "Then make a test request to see the actual error"
