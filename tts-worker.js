/* Pocket TTS ONNX WASM worker for the 4-card N5 test.
 * Pure browser inference: text_conditioner -> flow_lm_main -> flow_lm_flow -> mimi_decoder.
 * It consumes Python-prepared exact SentencePiece token IDs and voice embeddings.
 * NOTE: there is deliberately no server-side TTS API and no pre-recorded speech.
 */
'use strict';

const ORT_VERSION = '1.20.0';
const ORT_DIST = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
let ortReady = false;
let currentLanguage = null;
let engine = null;
let cancelGeneration = false;
let modelManifest = null;

function emit(type, detail = {}) { self.postMessage({ type, ...detail }); }

function tensor(type, data, dims) { return new ort.Tensor(type, data, dims); }
function f32(data, dims) { return tensor('float32', data, dims); }
function zeroFloat(dims, fill = 0) {
  const count = dims.reduce((a, b) => a * b, 1);
  const data = new Float32Array(count);
  if (fill !== 0) data.fill(fill);
  return f32(data, dims);
}
function makeState(manifest) {
  const state = {};
  for (const entry of manifest) {
    const dims = entry.shape;
    const count = dims.reduce((a, b) => a * b, 1);
    let data;
    switch (entry.dtype) {
      case 'float32': data = new Float32Array(count); break;
      case 'float16': data = new Uint16Array(count); break;
      case 'int64': data = new BigInt64Array(count); break;
      case 'bool': data = new Uint8Array(count); break;
      case 'int32': data = new Int32Array(count); break;
      default: throw new Error(`Dtype ONNX non previsto: ${entry.dtype}`);
    }
    if (entry.fill === 'ones') {
      if (entry.dtype === 'int64') data.fill(1n);
      else if (entry.dtype === 'float16') data.fill(0x3c00);
      else data.fill(1);
    } else if (entry.fill === 'nan') {
      if (entry.dtype === 'float16') data.fill(0x7e00);
      else if (entry.dtype === 'float32') data.fill(NaN);
      else throw new Error('Manifest richiede NaN in dtype non floating-point');
    }
    state[entry.input_name] = tensor(entry.dtype, data, dims);
  }
  return state;
}
function cloneState(state) {
  const copy = {};
  for (const [name, value] of Object.entries(state)) {
    copy[name] = tensor(value.type, value.data.slice(), value.dims.slice());
  }
  return copy;
}
function updateState(state, outputs, session, manifest, offset) {
  const names = session.outputNames;
  for (const entry of manifest) {
    const name = names[offset + entry.index];
    if (!name || !outputs[name]) {
      throw new Error(`Manifest output ${offset + entry.index} non valido per ${session.outputNames.join(',')}`);
    }
    state[entry.input_name] = outputs[name];
  }
}
function firstOutput(session, outputs, offset = 0) {
  const name = session.outputNames[offset];
  if (!name || !outputs[name]) throw new Error(`Output mancante, indice ${offset}`);
  return outputs[name];
}
function seeded(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function randNormal(prng) {
  const u = Math.max(prng(), 1e-10);
  const v = prng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

async function setupOrt() {
  if (ortReady) return;
  emit('status', { message: 'Caricamento ONNX Runtime Web (WASM)…' });
  try { importScripts('./vendor/ort.min.js'); }
  catch (_err) { importScripts(`${ORT_DIST}ort.min.js`); }
  if (typeof ort === 'undefined') throw new Error('ONNX Runtime Web non disponibile. Controlla la connessione CDN.');
  ort.env.wasm.numThreads = 1; // Safari iOS / GitHub Pages (no COOP/COEP)
  ort.env.wasm.simd = true;
  ort.env.wasm.proxy = false;
  ort.env.wasm.wasmPaths = ORT_DIST;
  ortReady = true;
}

async function dispose() {
  if (!engine) return;
  const sessions = engine.sessions;
  engine = null;
  currentLanguage = null;
  if (sessions) for (const session of Object.values(sessions)) {
    try { await session.release(); } catch (_e) {}
  }
}

async function getBinary(url) {
  const response = await fetch(url, { cache: 'default' });
  if (!response.ok) throw new Error(`Errore HTTP ${response.status} per ${url}. Avvia prepare_assets.py.`);
  return await response.arrayBuffer();
}

async function loadLanguage(lang, manifest) {
  if (lang === currentLanguage && engine) return;
  if (!manifest.languages?.[lang]) throw new Error(`Lingua non preparata: ${lang}`);
  await dispose(); // keep only one language in memory (important on iOS)
  await setupOrt();
  modelManifest = manifest;
  const cfg = manifest.languages[lang];
  const base = cfg.directory;
  emit('status', {message: `Caricamento modello ${lang === 'ja' ? 'giapponese' : 'italiano'}…`});
  const bundle = await (await fetch(base + 'bundle.json')).json();
  const modelOrder = ['text_conditioner', 'flow_lm_main', 'flow_lm_flow', 'mimi_decoder'];
  const sessions = {};
  try {
    for (let i = 0; i < modelOrder.length; i++) {
      const stem = modelOrder[i];
      const name = cfg.modelFiles[stem];
      if (!name) throw new Error(`Nessun modello per ${stem}`);
      emit('status', {message: `${lang.toUpperCase()}: ${i + 1}/${modelOrder.length} · ${name}…`});
      const bytes = await getBinary(base + name);
      sessions[stem] = await ort.InferenceSession.create(bytes, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
    }
    emit('status', {message: `${lang.toUpperCase()}: preparazione della voce…`});
    const voiceBytes = await getBinary(base + 'voice.f32');
    const voice = new Float32Array(voiceBytes);
    if (voice.length !== cfg.voiceFrames * cfg.voiceDim) {
      throw new Error(`Dimensione voce incongruente: ${voice.length} vs ${cfg.voiceFrames} x ${cfg.voiceDim}`);
    }
    const baseState = makeState(bundle.flow_lm_state_manifest);
    const voiceResult = await sessions.flow_lm_main.run({
      sequence: zeroFloat([1, 0, bundle.latent_dim]),
      text_embeddings: f32(voice, [1, cfg.voiceFrames, cfg.voiceDim]),
      ...baseState,
    });
    updateState(baseState, voiceResult, sessions.flow_lm_main, bundle.flow_lm_state_manifest, 2);
    engine = {lang, bundle, cfg, sessions, voiceState: baseState};
    currentLanguage = lang;
    emit('ready', {language: lang, sampleRate: bundle.sample_rate});
  } catch (err) {
    for (const session of Object.values(sessions)) {
      try { await session.release(); } catch (_e) {}
    }
    throw err;
  }
}

async function synthesizeTokens(tokenIds, options, onProgress) {
  if (!engine) throw new Error('Modello non caricato');
  const {bundle, sessions, voiceState} = engine;
  const state = cloneState(voiceState);
  const ids = BigInt64Array.from(tokenIds.map(x => BigInt(x)));
  const tokenResult = await sessions.text_conditioner.run({
    token_ids: tensor('int64', ids, [1, ids.length]),
  });
  const textEmbeddings = firstOutput(sessions.text_conditioner, tokenResult);
  let mainResult = await sessions.flow_lm_main.run({
    sequence: zeroFloat([1, 0, bundle.latent_dim]),
    text_embeddings: textEmbeddings,
    ...state,
  });
  updateState(state, mainResult, sessions.flow_lm_main, bundle.flow_lm_state_manifest, 2);

  const maxFrames = options.maxFrames;
  const lsdSteps = options.lsdSteps;
  const noiseStd = Math.sqrt(options.temperature);
  const random = seeded(options.seed);
  let curr = zeroFloat([1, 1, bundle.latent_dim], NaN);
  const emptyText = zeroFloat([1, 0, bundle.conditioning_dim]);
  const latents = [];
  let eosStep = null;
  let stopReason = 'frame_limit';
  for (let step = 0; step < maxFrames; step++) {
    if (cancelGeneration) throw new Error('Generazione annullata');
    mainResult = await sessions.flow_lm_main.run({
      sequence: curr,
      text_embeddings: emptyText,
      ...state,
    });
    const c = firstOutput(sessions.flow_lm_main, mainResult, 0);
    const eos = Number(firstOutput(sessions.flow_lm_main, mainResult, 1).data[0]);
    updateState(state, mainResult, sessions.flow_lm_main, bundle.flow_lm_state_manifest, 2);
    if (eosStep === null && eos > -4.0) eosStep = step;
    if (eosStep !== null && step >= eosStep + options.framesAfterEos) {
      stopReason = 'eos';
      break;
    }
    const x = new Float32Array(bundle.latent_dim);
    if (noiseStd > 0) for (let k = 0; k < x.length; k++) x[k] = randNormal(random) * noiseStd;
    for (let j = 0; j < lsdSteps; j++) {
      const dt = 1 / lsdSteps;
      const s = new Float32Array([j * dt]);
      const t = new Float32Array([(j + 1) * dt]);
      const flowResult = await sessions.flow_lm_flow.run({
        c,
        s: f32(s, [1, 1]),
        t: f32(t, [1, 1]),
        x: f32(x, [1, bundle.latent_dim]),
      });
      const delta = firstOutput(sessions.flow_lm_flow, flowResult).data;
      for (let k = 0; k < x.length; k++) x[k] += delta[k] * dt;
    }
    latents.push(x);
    curr = f32(x, [1, 1, bundle.latent_dim]);
    if (step % 5 === 0) onProgress(step + 1, maxFrames);
  }
  if (!latents.length) throw new Error('Generazione di zero frame: controllo EOS/voce necessario');
  const mimiState = makeState(bundle.mimi_state_manifest);
  const audioPieces = [];
  const decodeChunk = 15;
  for (let start = 0; start < latents.length; start += decodeChunk) {
    if (cancelGeneration) throw new Error('Generazione annullata');
    const count = Math.min(decodeChunk, latents.length - start);
    const lat = new Float32Array(count * bundle.latent_dim);
    for (let n = 0; n < count; n++) lat.set(latents[start + n], n * bundle.latent_dim);
    const decoded = await sessions.mimi_decoder.run({
      latent: f32(lat, [1, count, bundle.latent_dim]),
      ...mimiState,
    });
    const audio = firstOutput(sessions.mimi_decoder, decoded).data;
    audioPieces.push(Float32Array.from(audio));
    updateState(mimiState, decoded, sessions.mimi_decoder, bundle.mimi_state_manifest, 1);
  }
  const totalLength = audioPieces.reduce((s, x) => s + x.length, 0);
  const combined = new Float32Array(totalLength);
  let cursor = 0;
  for (const x of audioPieces) {combined.set(x, cursor);cursor += x.length;}
  return {samples: combined, sampleRate: bundle.sample_rate, frames: latents.length, eosStep, stopReason};
}

async function generate(request) {
  const started = performance.now();
  cancelGeneration = false;
  const opts = request.options;
  const sections = request.segments;
  const samples = [];
  let count = 0;
  const reports = [];
  // Real PCM silence *before* speech: helps isolate late playback initialization,
  // but cannot recover phonemes that the generative model never produced.
  const preSilenceMs = Math.max(0,Math.min(1500,Math.round(Number(opts.startSilenceMs) || 0)));
  const pauseMs = Math.max(0,Math.min(1500,Math.round(Number(opts.pauseMs) || 0)));
  const sampleRate = engine.bundle.sample_rate;
  if (preSilenceMs > 0) {
    const lead = new Float32Array(Math.round(preSilenceMs / 1000 * sampleRate));
    samples.push(lead);count += lead.length;
  }
  for (let i = 0; i < sections.length; i++) {
    if (cancelGeneration) throw new Error('Generazione annullata');
    const item = sections[i];
    emit('status', {message: `${request.language.toUpperCase()}: frase ${i + 1}/${sections.length}…`});
    const audio = await synthesizeTokens(item.tokens, {...opts, seed: opts.seed + i}, (frame, maxFrames) => {
      emit('progress', {requestId: request.requestId, frame, maxFrames, segment: i + 1});
    });
    samples.push(audio.samples);
    count += audio.samples.length;
    reports.push({text: item.text, tokenCount:item.tokens.length,frames:audio.frames,
                  eosStep: audio.eosStep,stopReason:audio.stopReason,
                  duration: audio.samples.length / audio.sampleRate});
    if (i < sections.length - 1) {
      const pad = new Float32Array(Math.round((pauseMs / 1000) * audio.sampleRate));
      samples.push(pad); count += pad.length;
    }
  }
  const result = new Float32Array(count);
  let offset = 0;
  for (const chunk of samples) {result.set(chunk, offset);offset += chunk.length;}
  emit('audio', {
    requestId: request.requestId, language: request.language,
    sampleRate: engine.bundle.sample_rate, samples: result,
    reports, generationSeconds: (performance.now()-started)/1000,
    pauseMs, startSilenceMs:preSilenceMs,
  });
}

self.onmessage = async (event) => {
  const message = event.data;
  if (message.type === 'cancel') {cancelGeneration = true;return;}
  try {
    if (message.type === 'load') await loadLanguage(message.language, message.manifest);
    else if (message.type === 'generate') {
      if (message.language !== currentLanguage || !engine) {
        await loadLanguage(message.language, message.manifest);
      }
      await generate(message);
    }
    else if (message.type === 'dispose') await dispose();
  } catch (err) {
    emit('error', {requestId: message.requestId ?? null, message:String(err.message || err),
                   stack: String(err.stack || '').slice(0, 1600)});
  }
};
