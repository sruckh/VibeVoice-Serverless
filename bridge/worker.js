/**
 * Cloudflare Worker: OpenAI TTS API → RunPod VibeVoice Bridge
 *
 * Translates OpenAI Text-to-Speech API requests to RunPod VibeVoice custom format
 * and returns OpenAI-compatible responses (raw audio bytes).
 */

// Cache for voice mappings (5 minute TTL)
let voiceMappingCache = null;
let lastFetch = 0;
const CACHE_TTL = 5 * 60 * 1000; // 5 minutes

export default {
  async fetch(request, env, ctx) {
    // CORS preflight
    if (request.method === 'OPTIONS') {
      return handleCORS();
    }

    // Only handle POST requests to /v1/audio/speech
    if (request.method !== 'POST') {
      return errorResponse('Method not allowed', 405);
    }

    const url = new URL(request.url);
    const path = url.pathname;

    // Route to streaming or batch
    if (path === '/api/tts/stream') {
      return handleStreamingTTS(request, env, ctx);
    }

    if (path !== '/v1/audio/speech') {
      return errorResponse('Not found. Use POST /v1/audio/speech or /api/tts/stream', 404);
    }

    // Authenticate request (if AUTH_TOKEN is configured)
    if (env.AUTH_TOKEN) {
      const authHeader = request.headers.get('Authorization');

      if (!authHeader) {
        return new Response(
          JSON.stringify({
            error: {
              message: 'Missing Authorization header',
              type: 'authentication_error',
              param: null,
              code: 'missing_authorization'
            }
          }),
          {
            status: 401,
            headers: {
              'Content-Type': 'application/json',
              'Access-Control-Allow-Origin': '*'
            }
          }
        );
      }

      const token = authHeader.replace(/^Bearer\s+/i, '');

      if (token !== env.AUTH_TOKEN) {
        return new Response(
          JSON.stringify({
            error: {
              message: 'Invalid authentication token',
              type: 'authentication_error',
              param: null,
              code: 'invalid_token'
            }
          }),
          {
            status: 401,
            headers: {
              'Content-Type': 'application/json',
              'Access-Control-Allow-Origin': '*'
            }
          }
        );
      }
    }

    try {
      // Parse OpenAI TTS request
      const openaiRequest = await request.json();

      // Validate required fields
      const { model, input, voice, response_format = 'mp3', speed = 1.0, stream = false } = openaiRequest;

      if (!model) {
        return openaiError('Missing required parameter: model', 'invalid_request_error', 'model');
      }
      if (!input) {
        return openaiError('Missing required parameter: input', 'invalid_request_error', 'input');
      }
      if (!voice) {
        return openaiError('Missing required parameter: voice', 'invalid_request_error', 'voice');
      }

      // Load voice mappings from R2
      const voiceMappings = await getVoiceMappings(env);

      // Resolve voice to speaker name
      const speakerName = voiceMappings[voice];
      if (!speakerName) {
        const available = Object.keys(voiceMappings).join(', ');
        return openaiError(
          `Invalid voice '${voice}'. Available voices: ${available}`,
          'invalid_request_error',
          'voice'
        );
      }

      // Warn about unsupported features (but don't error)
      if (response_format !== 'mp3' && response_format !== 'pcm') {
        console.warn(`Unsupported response_format: ${response_format}. Only 'mp3' or 'pcm' supported. Defaulting to mp3.`);
      }
      if (speed !== 1.0) {
        console.warn(`Speed parameter (${speed}) is not supported and will be ignored.`);
      }

      console.log(`OpenAI TTS request: voice=${voice} (${speakerName}), text_len=${input.length}, format=${response_format}, stream=${stream}`);

      if (stream) {
        return handleOpenAIStreaming(env, {
          text: input,
          speaker_name: speakerName,
          output_format: response_format === 'pcm' ? 'pcm_16' : 'mp3'
        });
      }

      if (!env.RUNPOD_URL) {
        return openaiError('RunPod endpoint not configured', 'server_error', null);
      }
      if (!env.RUNPOD_API_KEY) {
        return openaiError('RunPod API key not configured', 'server_error', null);
      }

      // BATCH MODE: Use streaming logic + accumulation to avoid RunPod payload limits
      // We force 'stream=true' in the RunPod request, even though this is a batch OpenAI request
      const runpodUrls = buildRunpodUrls(env.RUNPOD_URL);
      
      const audioBytes = await handleBatchViaStreaming({
        runpodUrls,
        apiKey: env.RUNPOD_API_KEY,
        text: input,
        speakerName,
        outputFormat: response_format === 'pcm' ? 'pcm_16' : 'mp3'
      });

      // Return raw audio bytes (OpenAI format)
      return new Response(audioBytes, {
        status: 200,
        headers: {
          'Content-Type': response_format === 'pcm' ? 'audio/pcm' : 'audio/mpeg',
          'Access-Control-Allow-Origin': '*',
          'Cache-Control': 'no-cache'
        }
      });

    } catch (error) {
      console.error('Worker error:', error);
      return openaiError(error.message, 'server_error', null);
    }
  }
};

/**
 * Handle streaming TTS requests from the frontend
 */
async function handleStreamingTTS(request, env, ctx) {
  const requestId = crypto.randomUUID();
  const startTime = Date.now();

  console.log(`[Tier 2][CF][${requestId}] Streaming TTS request received`);

  try {
    const { text, input, voice, service, response_format } = await request.json();
    const resolvedText = text || input;

    // Validation
    if (!resolvedText || resolvedText.trim().length === 0) {
      return errorResponse('Text is required', 400);
    }

    if (!voice) {
      return errorResponse('Voice is required', 400);
    }

    // Load voice mappings from R2
    const voiceMappings = await getVoiceMappings(env);

    // Resolve voice to speaker name
    const speakerName = voiceMappings[voice];
    if (!speakerName) {
      const available = Object.keys(voiceMappings).join(', ');
      return errorResponse(`Invalid voice '${voice}'. Available voices: ${available}`, 400);
    }

    // Get RunPod endpoint
    const runpodUrls = buildRunpodUrls(env.RUNPOD_URL);
    const apiKey = env.RUNPOD_API_KEY;

    if (!env.RUNPOD_URL) {
      return errorResponse('RunPod endpoint not configured', 500);
    }

    if (!apiKey) {
      return errorResponse('RunPod API key not configured', 500);
    }

    console.log(`[Tier 2][CF][${requestId}] Connecting to Tier 3 (RunPod) for voice=${voice} (${speakerName})...`);

    // Create transform stream for forwarding
    const { readable, writable } = new TransformStream();

    // Start forwarding in background
    ctx.waitUntil(forwardRunPodStream({
      runpodUrls,
      apiKey,
      text: resolvedText,
      speakerName,
      service,
      writable,
      requestId,
      startTime
    }));

    // Return streaming response immediately
    return new Response(readable, {
      headers: {
        'Content-Type': 'audio/octet-stream',
        'Cache-Control': 'no-cache, no-transform',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization'
      }
    });

  } catch (error) {
    console.error(`[Tier 2][CF] Error:`, error);
    return errorResponse(error.message, 500);
  }
}

/**
 * Execute a batch request by streaming and accumulating chunks
 */
async function handleBatchViaStreaming({ runpodUrls, apiKey, text, speakerName, outputFormat }) {
  const requestId = crypto.randomUUID();
  console.log(`[Tier 2][CF][${requestId}] Starting batch-via-streaming for ${text.length} chars`);

  // Submit job
  const runpodResponse = await fetch(runpodUrls.run, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`
    },
    body: JSON.stringify({
      input: {
        text,
        speaker_name: speakerName,
        stream: true, // Always stream from RunPod to avoid payload limits
        output_format: outputFormat
      }
    })
  });

  if (!runpodResponse.ok) {
    const errorText = await runpodResponse.text();
    throw new Error(`RunPod submit failed: ${runpodResponse.status} - ${errorText}`);
  }

  const jobData = await runpodResponse.json();
  const jobId = extractRunpodJobId(jobData);
  if (!jobId) {
    throw new Error(`RunPod did not return a job id: ${JSON.stringify(jobData)}`);
  }

  // Accumulator
  const chunks = [];
  const writer = {
    write(chunk) {
      chunks.push(chunk);
      return Promise.resolve();
    },
    close() { return Promise.resolve(); }
  };

  // Poll and accumulate
  await pollRunpodStream({
    runpodUrls,
    apiKey,
    jobId,
    writer,
    requestId
  });

  // Concatenate chunks
  const totalLength = chunks.reduce((acc, chunk) => acc + chunk.length, 0);
  const result = new Uint8Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.length;
  }

  console.log(`[Tier 2][CF][${requestId}] Batch complete: ${totalLength} bytes assembled`);
  return result.buffer;
}

/**
 * Forward RunPod stream to client (Tier 1)
 */
async function forwardRunPodStream({ runpodUrls, apiKey, text, speakerName, service, writable, requestId, startTime }) {
  const writer = writable.getWriter();

  try {
    // Submit async job to Tier 3 (RunPod)
    const runpodResponse = await fetch(runpodUrls.run, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`
      },
      body: JSON.stringify({
        input: {
          text,
          speaker_name: speakerName,
          service,
          stream: true,
          output_format: 'pcm_16'
        }
      })
    });

    if (!runpodResponse.ok) {
      const errorText = await runpodResponse.text();
      throw new Error(`Tier 3 error: ${runpodResponse.status} - ${errorText}`);
    }

    const runpodResult = await runpodResponse.json();
    const jobId = extractRunpodJobId(runpodResult);
    if (!jobId) {
      throw new Error(`RunPod did not return a job id: ${JSON.stringify(runpodResult)}`);
    }

    console.log(`[Tier 2][CF][${requestId}] Tier 3 accepted job ${jobId}, streaming...`);

    const { totalChunks, totalBytes } = await pollRunpodStream({
      runpodUrls,
      apiKey,
      jobId,
      writer,
      requestId
    });

    const elapsed = Date.now() - startTime;
    console.log(`[Tier 2][CF][${requestId}] Stream complete: ${totalChunks} chunks, ${totalBytes} bytes, ${elapsed}ms`);

  } catch (error) {
    console.error(`[Tier 2][CF][${requestId}] Forward error:`, error);
  } finally {
    await writer.close();
  }
}

async function handleOpenAIStreaming(env, params) {
  const { text, speaker_name, output_format } = params;
  const requestId = crypto.randomUUID();

  const runpodUrls = buildRunpodUrls(env.RUNPOD_URL);

  const runResponse = await fetch(runpodUrls.run, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${env.RUNPOD_API_KEY}`
    },
    body: JSON.stringify({
      input: {
        text,
        speaker_name,
        stream: true,
        output_format
      }
    })
  });

  if (!runResponse.ok) {
    const errorText = await runResponse.text();
    throw new Error(`RunPod submit failed: ${runResponse.status} - ${errorText}`);
  }

  const jobData = await runResponse.json();
  const jobId = extractRunpodJobId(jobData);
  if (!jobId) {
    throw new Error(`RunPod did not return a job id: ${JSON.stringify(jobData)}`);
  }

  const { readable, writable } = new TransformStream();
  const writer = writable.getWriter();

  (async () => {
    try {
      await pollRunpodStream({
        runpodUrls,
        apiKey: env.RUNPOD_API_KEY,
        jobId,
        writer,
        requestId
      });
    } catch (e) {
      console.error(`[Tier 2][CF][${requestId}] Streaming error:`, e);
    } finally {
      await writer.close();
    }
  })();

  return new Response(readable, {
    headers: {
      'Content-Type': output_format === 'mp3' ? 'audio/mpeg' : 'audio/pcm',
      'Transfer-Encoding': 'chunked',
      'Cache-Control': 'no-cache',
      'X-Accel-Buffering': 'no'
    }
  });
}

function buildRunpodUrls(runpodUrl) {
  const base = runpodUrl.replace(/\/(runsync|run|status)(\/.*)?$/, '');
  return {
    runsync: `${base}/runsync`,
    run: `${base}/run`,
    status: `${base}/status`,
    stream: `${base}/stream`
  };
}

function extractRunpodJobId(runpodResult) {
  return runpodResult.id || runpodResult.jobId || runpodResult.job_id;
}

async function callRunpodRunsync({ runpodUrls, apiKey, runpodRequest }) {
  const runpodResponse = await fetch(runpodUrls.runsync, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`
    },
    body: JSON.stringify(runpodRequest)
  });

  if (!runpodResponse.ok) {
    const errorText = await runpodResponse.text();
    console.error('RunPod error:', errorText);
    throw new Error(`RunPod service error: ${runpodResponse.status} ${runpodResponse.statusText}`);
  }

  const runpodResult = await runpodResponse.json();

  if (runpodResult.status === 'IN_PROGRESS' && runpodResult.id) {
    return pollRunpodStatus({ runpodUrls, apiKey, jobId: runpodResult.id });
  }

  if (runpodResult.status === 'FAILED') {
    throw new Error(runpodResult.error || 'RunPod job failed');
  }

  return runpodResult;
}

async function pollRunpodStatus({ runpodUrls, apiKey, jobId }) {
  let backoffMs = 500;
  const maxBackoffMs = 10000;

  while (true) {
    await sleep(backoffMs);

    const statusResponse = await fetch(`${runpodUrls.status}/${jobId}`, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${apiKey}`
      }
    });

    if (!statusResponse.ok) {
      const errorText = await statusResponse.text();
      throw new Error(`RunPod status error: ${statusResponse.status} - ${errorText}`);
    }

    const statusResult = await statusResponse.json();

    if (statusResult.status === 'COMPLETED') {
      return statusResult;
    }

    if (statusResult.status === 'FAILED') {
      throw new Error(statusResult.error || 'RunPod job failed');
    }

    backoffMs = Math.min(backoffMs * 2, maxBackoffMs);
  }
}

async function pollRunpodStream({ runpodUrls, apiKey, jobId, writer, requestId }) {
  let totalChunks = 0;
  let totalBytes = 0;
  let maxChunkProcessed = 0; // Track by ID instead of array index
  let pollInterval = 500;
  let isFinished = false;
  const startTime = Date.now();
  const timeoutMs = 300000;
  
  // Safety mechanism: if RunPod says COMPLETED but we haven't seen our "complete" message,
  // we keep polling for a limited time to drain the stream.
  let completedAt = null;
  const DRAIN_TIMEOUT_MS = 10000; // Wait up to 10s for stream to catch up

  while (!isFinished && (Date.now() - startTime) < timeoutMs) {
    const streamResponse = await fetch(`${runpodUrls.stream}/${jobId}`, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${apiKey}`
      }
    });

    if (!streamResponse.ok) {
      const errorText = await streamResponse.text();
      throw new Error(`RunPod stream error: ${streamResponse.status} - ${errorText}`);
    }

    const data = await streamResponse.json();
    const streamData = data.stream || [];

    // Iterate through ALL available items to check for new chunks
    // This is robust against both accumulated lists and partial updates
    for (const rawItem of streamData) {
      // RunPod sometimes wraps the yield in an 'output' property
      const item = rawItem.output || rawItem;

      if (item.status === 'streaming' && item.audio_chunk) {
        // Only process if we haven't seen this chunk number yet
        // If chunk ID is missing (shouldn't happen), fall back to processing
        const chunkId = item.chunk || (maxChunkProcessed + 1);
        
        if (chunkId > maxChunkProcessed) {
          console.log(`[Tier 2][CF][${requestId}] Processing chunk ${chunkId}`);
          const audioData = base64ToArrayBuffer(item.audio_chunk);
          await writer.write(new Uint8Array(audioData));
          totalChunks += 1;
          totalBytes += audioData.byteLength;
          maxChunkProcessed = chunkId;
          
          // Reset poll interval on activity
          pollInterval = 500;
        }
      } else if (item.status === 'complete') {
        if (!isFinished) {
          console.log(`[Tier 2][CF][${requestId}] RunPod signaled completion via stream`);
          isFinished = true;
        }
      } else if (item.error) {
        console.error(`[Tier 2][CF][${requestId}] RunPod returned error in stream:`, item.error);
      }
    }

    // Adaptive polling: slow down if no new data
    pollInterval = Math.min(pollInterval * 1.5, 5000);

    // Check Job Status
    if (data.status === 'FAILED') {
      console.error(`[Tier 2][CF][${requestId}] RunPod job failed:`, data.error);
      isFinished = true;
    } else if (data.status === 'COMPLETED') {
      // If we haven't seen the stream completion message yet, start the drain timer
      if (!isFinished) {
        if (!completedAt) {
          console.log(`[Tier 2][CF][${requestId}] RunPod status is COMPLETED, waiting for stream drain...`);
          completedAt = Date.now();
          pollInterval = 200; // Poll faster to finish up
        } else if ((Date.now() - completedAt) > DRAIN_TIMEOUT_MS) {
            console.warn(`[Tier 2][CF][${requestId}] Timed out waiting for stream drain`);
            isFinished = true;
        }
      }
    }

    if (!isFinished) {
      await sleep(pollInterval);
    }
  }

  return { totalChunks, totalBytes };
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Load voice mappings from R2 bucket with caching
 */
async function getVoiceMappings(env) {
  const now = Date.now();

  if (voiceMappingCache && (now - lastFetch) < CACHE_TTL) {
    return voiceMappingCache;
  }

  try {
    const object = await env.VIBEVOICE_BUCKET.get('voices.json');

    if (!object) {
      throw new Error('voices.json not found in R2 bucket');
    }

    voiceMappingCache = await object.json();
    lastFetch = now;

    console.log('Loaded voice mappings:', Object.keys(voiceMappingCache));
    return voiceMappingCache;

  } catch (error) {
    console.error('Failed to load voice mappings:', error);

    if (voiceMappingCache) {
      console.warn('Using stale voice mapping cache due to error');
      return voiceMappingCache;
    }

    throw new Error(`Failed to load voice mappings: ${error.message}`);
  }
}

/**
 * Return OpenAI-formatted error response
 */
function openaiError(message, type, param) {
  return new Response(
    JSON.stringify({
      error: {
        message: message,
        type: type,
        param: param,
        code: null
      }
    }),
    {
      status: 400,
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*'
      }
    }
  );
}

/**
 * Generic error response
 */
function errorResponse(message, status = 400) {
  return new Response(
    JSON.stringify({ error: message }),
    {
      status: status,
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*'
      }
    }
  );
}

/**
 * Handle CORS preflight
 */
function handleCORS() {
  return new Response(null, {
    status: 204,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
      'Access-Control-Max-Age': '86400'
    }
  });
}

/**
 * Convert base64 string to ArrayBuffer
 */
function base64ToArrayBuffer(base64) {
  const binaryString = atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes.buffer;
}
