'use strict';
const $ = (id) => document.getElementById(id);
let manifest = null;
let worker = null;
let readyLanguage = null;
let busy = false;
let nextRequest = 0;
let requestPending = null;
let currentAudioUrl = null;
let currentCards = [];
const audioCache = new Map();

function status(message, state = '') {
  $('status').textContent = message;
  $('statusDot').className = `status-dot ${state}`;
}
function setBusy(value) {
  busy = value;
  document.querySelectorAll('[data-action], #loadJa, #loadIt').forEach(button => button.disabled = value || !manifest);
  $('stop').disabled = !value;
}
function updatePills() {
  for (const lang of ['ja','it']) {
    const el = $(lang === 'ja' ? 'jaState' : 'itState');
    el.classList.toggle('active',readyLanguage === lang);
    el.textContent = `${lang.toUpperCase()} · ${readyLanguage === lang ? 'pronto' : 'non caricato'}`;
  }
}
function wavFile(audio, sampleRate) {
  const n = audio.length;
  const b = new ArrayBuffer(44 + n * 2);
  const v = new DataView(b);
  function str(offset, s) {for (let i=0;i<s.length;i++) v.setUint8(offset+i,s.charCodeAt(i));}
  str(0,'RIFF'); v.setUint32(4,36+n*2,true);str(8,'WAVE');str(12,'fmt ');
  v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);
  v.setUint32(24,sampleRate,true);v.setUint32(28,sampleRate*2,true);
  v.setUint16(32,2,true);v.setUint16(34,16,true);
  str(36,'data');v.setUint32(40,n*2,true);
  for (let i=0;i<n;i++) {
    const x = Math.max(-1,Math.min(1,audio[i]));
    v.setInt16(44+i*2,x<0?Math.round(x*32768):Math.round(x*32767),true);
  }
  return new Blob([b],{type:'audio/wav'});
}
function showAudio(blob, label) {
  if (currentAudioUrl) URL.revokeObjectURL(currentAudioUrl);
  currentAudioUrl = URL.createObjectURL(blob);
  $('audio').src = currentAudioUrl;
  $('download').href = currentAudioUrl;
  $('download').download = `kotoba_onnx_${label}.wav`;
  $('download').classList.remove('disabled');
  $('download').removeAttribute('aria-disabled');
  // iOS may block autoplay after slow async inference; native audio control remains usable.
  $('audio').play().catch(() => status('Audio pronto: premi Play nel lettore.', 'ready'));
}
function summary(result) {
  const speechDuration = result.reports.reduce((a,r)=>a+r.duration,0);
  const audioDuration = (result.sampleCount ?? Math.round(speechDuration * result.sampleRate)) / result.sampleRate;
  const reason = result.reports.map(r=>`${r.stopReason}${r.eosStep===null?'':`@${r.eosStep}`}`).join(', ');
  $('metrics').textContent = `${result.reports.length} segmento/i · ${speechDuration.toFixed(2)} s parlati / ${audioDuration.toFixed(2)} s WAV · ${result.generationSeconds.toFixed(1)} s calcolo · RTF ${(result.generationSeconds / Math.max(speechDuration,.001)).toFixed(2)} · pausa ${result.pauseMs} ms · pre-silenzio ${result.startSilenceMs} ms · stop: ${reason}`;
}
function ensureWorker() {
  if (worker) return;
  worker = new Worker('./tts-worker.js');
  worker.onmessage = async ({data:msg}) => {
    if (msg.type === 'status') status(msg.message,'loading');
    else if (msg.type === 'progress') {
      status(`${msg.segment} · frame ${msg.frame}/${msg.maxFrames} · generazione ONNX…`,'loading');
    }
    else if (msg.type === 'ready') {
      readyLanguage = msg.language;
      updatePills();
      status(`Modello ${msg.language.toUpperCase()} pronto (${msg.sampleRate} Hz)`,'ready');
      if (requestPending?.kind === 'load' && requestPending.language === msg.language) {
        requestPending = null;
        setBusy(false);
      }
    }
    else if (msg.type === 'audio') {
      if (requestPending?.id !== msg.requestId) return;
      const label = requestPending.label;
      const b = wavFile(msg.samples,msg.sampleRate);
      audioCache.set(requestPending.cacheKey, {blob:b,summary:{reports:msg.reports,generationSeconds:msg.generationSeconds,
        sampleCount:msg.samples.length,sampleRate:msg.sampleRate,pauseMs:msg.pauseMs,startSilenceMs:msg.startSilenceMs}});
      showAudio(b,label);
      summary(msg);
      status(`Audio ${msg.language.toUpperCase()} generato nel browser.`, 'ready');
      requestPending = null;
      setBusy(false);
    }
    else if (msg.type === 'error') {
      console.error('PocketTTS ONNX error:',msg.message,msg.stack);
      status(`Errore: ${msg.message}`, 'error');
      requestPending = null;
      setBusy(false);
    }
  };
  worker.onerror = (event) => {
    console.error('Worker:',event.message);
    status(`Errore JavaScript/ONNX Worker: ${event.message || 'vedi console'}`,'error');
    readyLanguage = null;
    requestPending = null;
    setBusy(false);
  };
}
function numericMs(id, fallback) {
  const value = Number($(id).value);
  return Number.isFinite(value) ? Math.max(0,Math.min(1500,Math.round(value))) : fallback;
}
function options() {
  return {temperature: Number($('temp').value), lsdSteps: Number($('steps').value),
    maxFrames:Number($('maxFrames').value), framesAfterEos:Number($('afterEos').value),
    pauseMs:numericMs('pauseMs',250), startSilenceMs:numericMs('startSilenceMs',200), seed:1234};
}
function load(lang) {
  if (!manifest || busy) return;
  if (lang === readyLanguage) {status(`Modello ${lang.toUpperCase()} già pronto`, 'ready');return;}
  ensureWorker();
  readyLanguage = null;
  updatePills();
  setBusy(true);
  requestPending = {kind:'load',language:lang};
  status(`Caricamento ${lang.toUpperCase()}…`,'loading');
  worker.postMessage({type:'load',language:lang,manifest});
}
function play(cardIndex, partName) {
  if (!manifest || busy) return;
  const card=manifest.cards[cardIndex];
  const lang=partName.startsWith('ja') ? 'ja' : 'it';
  const mode = $('itMode').value;
  const selectedPart = partName === 'itMeaning' && mode === 'context' && card.tts.itMeaningContext
    ? 'itMeaningContext' : partName;
  const segments = card.tts[selectedPart].segments;
  const opts=options();
  const cacheKey = JSON.stringify([card.id,selectedPart,opts]);
  const cached=audioCache.get(cacheKey);
  if (cached) {
    showAudio(cached.blob,`${card.jp}_${partName}`);
    summary(cached.summary);
    status('Audio già generato: riutilizzo dalla memoria del browser.', 'ready');
    return;
  }
  ensureWorker();
  setBusy(true);
  const id=++nextRequest;
  requestPending={kind:'generate',id,cacheKey,label:`${card.id}_${partName}`};
  status(`Avvio sintesi ${lang.toUpperCase()} per ${card.jp}${selectedPart === 'itMeaningContext' ? ' (significato contestualizzato)' : ''}…`,'loading');
  worker.postMessage({type:'generate',requestId:id,language:lang,manifest,
                      segments, options:opts});
}
function stop() {
  $('audio').pause();
  if (worker) worker.postMessage({type:'cancel'});
  if (requestPending?.kind === 'generate') {
    status('Annullamento richiesto…', 'loading');
  }
}
function textNode(tag, text, cls='') {
  const n=document.createElement(tag);
  if(cls)n.className=cls;
  n.textContent=text;
  return n;
}
function filteredCards() {
  const query = $('cardSearch').value.trim().toLocaleLowerCase();
  const limit = $('deckSize').value;
  // For the sample mode, limit the source before filtering, so 4 stays stable.
  const candidates = limit === 'all' ? manifest.cards : manifest.cards.slice(0,Number(limit));
  if (!query) return candidates;
  return candidates.filter(c=>[c.jp,c.kana,c.meaning,c.example.jp,c.example.it]
    .some(value=>String(value || '').toLocaleLowerCase().includes(query)));
}
function render() {
  if (!manifest) return;
  $('cards').replaceChildren();
  currentCards = filteredCards();
  currentCards.forEach((c,idx)=>{
    const originalIndex = manifest.cards.indexOf(c);
    const article=document.createElement('article');article.className='word-card';
    article.append(textNode('div',`CARD ${String(originalIndex+1).padStart(2,'0')} / ${String(manifest.cards.length).padStart(2,'0')}`, 'card-number'));
    const head=document.createElement('div');head.className='word-top';
    const word=document.createElement('div');
    word.append(textNode('div',c.jp,'word-jp'),textNode('div',c.kana,'kana'));
    head.append(word,textNode('span',c.pos || 'N5','pos')); article.append(head);
    article.append(textNode('div',c.meaning,'meaning'));
    const forms=document.createElement('div');forms.className='forms';
    for(const f of (c.forms||[]).slice(0,4)) forms.append(textNode('span',f.jp,'form'));
    article.append(forms,textNode('div','','divider'));
    const example=document.createElement('div');
    example.append(textNode('div','Esempio dal deck','example-label'));
    const exampleJa=textNode('div',c.example.jp,'example-jp');exampleJa.lang='ja';
    example.append(exampleJa,textNode('div',c.example.it,'example-it'));
    article.append(example);
    const actions=document.createElement('div'); actions.className='actions';
    for (const [part,label,cls] of [
      ['jaWord','▶ JA · Parola',''],['itMeaning','▶ IT · Significato','it'],
      ['jaExample','▶ JA · Esempio',''],['itExample','▶ IT · Traduzione','it']
    ]){
      const b=textNode('button',label,`play-btn ${cls}`);
      b.type='button';b.dataset.action=part;b.disabled=busy;
      b.addEventListener('click',()=>play(originalIndex,part));actions.append(b);
    }
    article.append(actions);$('cards').append(article);
  });
  if (!currentCards.length) $('cards').append(textNode('p','Nessuna card corrisponde alla ricerca.'));
  $('cardCount').textContent=`${currentCards.length} / ${manifest.cards.length} card N5`;
}
async function init() {
  $('temp').addEventListener('input',()=>{$('tempLabel').textContent=Number($('temp').value).toFixed(2).replace('.',',');});
  $('deckSize').addEventListener('change',render);
  $('cardSearch').addEventListener('input',render);
  for (const id of ['pauseMs','startSilenceMs']) {
    $(id).addEventListener('change',()=>{$(id).value=String(numericMs(id,id==='pauseMs'?250:200));});
  }
  $('loadJa').addEventListener('click',()=>load('ja'));
  $('loadIt').addEventListener('click',()=>load('it'));
  $('stop').addEventListener('click',stop);
  try {
    const r=await fetch('./assets/manifest.json',{cache:'no-store'});
    if(!r.ok) throw new Error('assets/manifest.json mancante: esegui prepare_assets.py prima di aprire la pagina');
    manifest=await r.json();
    if (!manifest.cards?.length || !manifest.languages?.ja || !manifest.languages?.it) {
      throw new Error('Manifest non valido: servono almeno una card e due lingue.');
    }
    $('heroCount').textContent=String(manifest.cards.length).padStart(2,'0');
    $('pauseMs').value=String(manifest.configuration?.pauseMs ?? 250);
    $('startSilenceMs').value=String(manifest.configuration?.startSilenceMs ?? 200);
    if (manifest.languages.it.voiceSharedWithJapanese === true) {
      $('voiceHint').textContent += ` Attenzione: voce italiana e giapponese usano entrambe ${manifest.languages.it.voiceSource}; per valutare l'italiano usa un WAV di riferimento italiano con -ItalianVoice.`;
    }
    render();
    setBusy(false);
    updatePills();
    status('Pronto. Premi un pulsante audio: il modello necessario verrà caricato.', 'ready');
  } catch (e) {
    status(e.message,'error');
    console.error(e);
    const msg=document.createElement('p');msg.style.color='#a34942';
    msg.textContent='I modelli non sono inclusi nello ZIP. Dalla cartella kotoba_japanese_onnx esegui prepare_assets.py con i bundle locali; vedi README.md.';
    $('cards').replaceChildren(msg);
  }
}
init();
