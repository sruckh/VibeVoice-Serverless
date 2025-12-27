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
    if (url.pathname !== '/v1/audio/speech') {
      return errorResponse('Not found. Use POST /v1/audio/speech', 404);
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
      const { model, input, voice, response_format = 'mp3', speed = 1.0 } = openaiRequest;

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
      if (response_format !== 'mp3') {
        console.warn(`Unsupported response_format: ${response_format}. Only 'mp3' is supported. Defaulting to mp3.`);
      }
      if (speed !== 1.0) {
        console.warn(`Speed parameter (${speed}) is not supported and will be ignored.`);
      }

      console.log(`OpenAI TTS request: voice=${voice} (${speakerName}), text_len=${input.length}, format=${response_format}`);

      // Translate to RunPod custom format
      const runpodRequest = {
        input: {
          text: input,
          speaker_name: speakerName,
          cfg_scale: 1.3,
          disable_prefill: false
        }
      };

      // Call RunPod serverless
      const runpodResponse = await fetch(env.RUNPOD_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${env.RUNPOD_API_KEY}`
        },
        body: JSON.stringify(runpodRequest)
      });

      if (!runpodResponse.ok) {
        const errorText = await runpodResponse.text();
        console.error('RunPod error:', errorText);
        return openaiError(
          `RunPod service error: ${runpodResponse.status} ${runpodResponse.statusText}`,
          'server_error',
          null
        );
      }

      const runpodResult = await runpodResponse.json();

      // Extract output from RunPod response
      const output = runpodResult.output || runpodResult;

      // Check for errors in response
      if (output.error || runpodResult.error) {
        const error = output.error || runpodResult.error;
        console.error('RunPod returned error:', error);
        return openaiError(error, 'server_error', null);
      }

      // Extract audio data (S3 URL or base64)
      let audioBytes;
      let contentType = 'audio/mpeg';

      if (output.audio_url) {
        // Fetch audio from S3 URL
        console.log('Fetching audio from S3:', output.audio_url);
        const s3Response = await fetch(output.audio_url);

        if (!s3Response.ok) {
          console.error('Failed to fetch from S3:', s3Response.status);
          return openaiError('Failed to fetch audio from S3', 'server_error', null);
        }

        audioBytes = await s3Response.arrayBuffer();
        contentType = s3Response.headers.get('Content-Type') || 'audio/mpeg';
        console.log('S3 Content-Type:', contentType);

      } else if (output.audio_base64 || output.audio) {
        // Decode base64 to binary
        const audioBase64 = output.audio_base64 || output.audio;
        audioBytes = base64ToArrayBuffer(audioBase64);

      } else {
        console.error('No audio data in RunPod response:', runpodResult);
        return openaiError('No audio data returned from RunPod', 'server_error', null);
      }

      // Return raw audio bytes (OpenAI format)
      return new Response(audioBytes, {
        status: 200,
        headers: {
          'Content-Type': contentType,
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
