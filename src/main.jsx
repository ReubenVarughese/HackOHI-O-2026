import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import './styles.css';
import sparkyFriendlyUrl from '../assets/sparky/Sparky 1.png';
import sparkyIdeaUrl from '../assets/sparky/Sparky 2.png';
import sparkyHighlightUrl from '../assets/sparky/Sparky 3.png';
import sparkyWarningUrl from '../assets/sparky/Sparky 4.png';

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL
  || (import.meta.env.PROD ? 'https://cspvisj22u.us-east-2.awsapprunner.com' : 'http://127.0.0.1:8000');

const contexts = [
  { id: 'OH-7C4A-118', area: 'North Columbus', circuit: 'Circuit 27', device: 'Recloser 27-E', exposure: 68, customers: 1840, latitude: 40.058, longitude: -82.998, source: 'Matched demonstration utility data', reclosers: 1, switches: 4, transformers: 12, topologyDepth: 4 },
  { id: 'OH-2F91-064', area: 'Westerville', circuit: 'Circuit 14', device: 'Fuse bank 14-B', exposure: 42, customers: 620, latitude: 40.126, longitude: -82.930, source: 'Matched demonstration utility data', fuses: 1, switches: 2, transformers: 8, topologyDepth: 3 },
  { id: 'OH-9D20-332', area: 'Gahanna', circuit: 'Circuit 31', device: 'Recloser 31-A', exposure: 79, customers: 2310, latitude: 40.018, longitude: -82.879, source: 'Matched demonstration utility data', reclosers: 1, switches: 5, transformers: 15, topologyDepth: 5 }
];

const stormProfiles = {
  moderate: { label: 'Moderate storm', risk: 0, duration: 0, description: 'Sustained wind 35–45 mph · isolated tree contact' },
  severe: { label: 'Severe storm', risk: 13, duration: 8, description: 'Sustained wind 50–65 mph · widespread tree contact' },
  extreme: { label: 'Extreme storm', risk: 24, duration: 18, description: 'Wind 65+ mph · access and mutual-aid constraints likely' }
};

function sparkyAvatar(text, messageId) {
  if (messageId === 'welcome-message') return sparkyFriendlyUrl;
  const lower = text.toLowerCase();
  if (lower.includes('checking') || lower.includes('risk') || lower.includes('alert')) return sparkyWarningUrl;
  if (lower.includes('prepare') || lower.includes('data') || lower.includes('support')) return sparkyIdeaUrl;
  if (lower.includes('restore') || lower.includes('return') || lower.includes('when')) return sparkyHighlightUrl;
  return sparkyFriendlyUrl;
}

function calculate(context, storm, damage, crews, access, priority, weather) {
  const profile = stormProfiles[storm];
  const weatherRisk = weatherSignal(weather);
  const risk = Math.min(96, Math.round(22 + context.exposure * 0.38 + profile.risk + weatherRisk.risk + damage * 4 - (access - 50) * 0.12));
  const low = Math.max(1, Math.round(3 + damage * 1.6 + profile.duration + weatherRisk.duration - crews * 1.2 + (100 - access) * 0.04));
  const high = Math.max(low + 2, Math.round(low + 7 + damage * 1.2 + weatherRisk.duration + (100 - access) * 0.08 + (priority ? 0 : 2)));
  const baseConfidence = Math.round(91 - damage * 2 - (100 - access) * 0.12);
  return { risk, low, high, eligible: risk >= 55 || priority, confidence: context.dataMode === 'fallback' ? Math.max(35, baseConfidence - 24) : Math.max(56, baseConfidence) };
}

function weatherSignal(weather) {
  if (!weather?.available) return { risk: 0, duration: 0 };
  const windRisk = weather.windSpeed >= 65 ? 12 : weather.windSpeed >= 45 ? 7 : weather.windSpeed >= 30 ? 3 : 0;
  const precipitationRisk = weather.precipitation >= 10 ? 5 : weather.precipitation >= 2 ? 2 : 0;
  return { risk: windRisk + precipitationRisk, duration: weather.windSpeed >= 45 ? 3 : 0 };
}

function weatherDescription(weather) {
  if (weather.loading) return 'Loading live weather…';
  if (weather.error) return 'Live weather unavailable · using selected storm outlook';
  if (!weather.available) return 'No live weather signal';
  return `${Math.round(weather.temperature)}°C · wind ${Math.round(weather.windSpeed)} km/h · rain ${weather.precipitation.toFixed(1)} mm`;
}

function preparationGuidance(storm, risk) {
  if (storm === 'extreme' || risk >= 70) {
    return {
      title: 'Prepare for an extended outage',
      items: ['Charge devices and gather water, food, medication, and lighting.', 'Use generators only outdoors and away from windows.', 'Secure loose outdoor items and avoid downed lines or flooded areas.', 'Keep an emergency communication plan and follow official instructions.']
    };
  }
  if (storm === 'severe' || risk >= 45) {
    return {
      title: 'Prepare for a possible outage',
      items: ['Prepare water, food, medication, and lighting for 24–48 hours.', 'Charge essential devices and unplug sensitive electronics.', 'Secure outdoor items and keep vehicles fueled.', 'Monitor official safety and utility updates.']
    };
  }
  return {
    title: 'Make a simple plan',
    items: ['Charge essential devices.', 'Keep medication and lighting ready.', 'Monitor official safety and utility updates.']
  };
}

function resolveLocation(latitude, longitude) {
  return {
    id: 'GENERIC-LOCATION',
    area: 'User-supplied location',
    circuit: 'Utility match unavailable',
    device: 'Protective-device data unavailable',
    exposure: 50,
    customers: 0,
    latitude,
    longitude,
    source: 'Utility evidence will be checked for this location',
    matchDistance: null,
    dataMode: 'fallback'
  };
}

function App() {
  const [showWelcome, setShowWelcome] = useState(true);
  const [welcomeLeaving, setWelcomeLeaving] = useState(false);
  const [contextId, setContextId] = useState('none');
  const [storm, setStorm] = useState('severe');
  const [damage, setDamage] = useState(4);
  const [crews, setCrews] = useState(3);
  const [access, setAccess] = useState(78);
  const [priority, setPriority] = useState(false);
  const [latitude, setLatitude] = useState('');
  const [longitude, setLongitude] = useState('');
  const [locationQuery, setLocationQuery] = useState('');
  const [locationName, setLocationName] = useState('');
  const [locationStatus, setLocationStatus] = useState('');
  const [activeTab, setActiveTab] = useState('overview');
  const [weather, setWeather] = useState({ loading: true, available: false, error: false });
  const [apiPrediction, setApiPrediction] = useState(null);
  const selectedContext = contexts.find((item) => item.id === contextId);
  const context = contextId === 'none'
    ? { id: 'NO-LOCATION', area: 'No location selected', circuit: 'Location required', device: 'Location required', exposure: 0, customers: 0, latitude: null, longitude: null, source: 'Enter a city and state to begin', dataMode: 'unselected' }
    : contextId === 'custom'
      ? { ...resolveLocation(Number(latitude), Number(longitude)), area: locationName || 'User-supplied location' }
      : selectedContext;
  const localResult = useMemo(() => calculate(context, storm, damage, crews, access, priority, weather), [context, storm, damage, crews, access, priority, weather]);
  const result = apiPrediction ? { ...localResult, risk: apiPrediction.risk, low: apiPrediction.low, high: apiPrediction.high, confidence: apiPrediction.confidence, eligible: apiPrediction.risk >= 55 || priority } : localResult;
  const riskLabel = result.risk >= 70 ? 'High' : result.risk >= 45 ? 'Moderate' : 'Low';
  useEffect(() => {
    const controller = new AbortController();
    if (contextId === 'none') {
      setWeather({ loading: false, available: false, error: false });
      return () => controller.abort();
    }
    const lat = Number(context.latitude);
    const lon = Number(context.longitude);
    if (!Number.isFinite(lat) || !Number.isFinite(lon) || lat < -90 || lat > 90 || lon < -180 || lon > 180) {
      setWeather({ loading: false, available: false, error: true });
      return () => controller.abort();
    }
    setWeather({ loading: true, available: false, error: false });
    const params = new URLSearchParams({
      latitude: String(lat),
      longitude: String(lon),
      current: 'temperature_2m,wind_speed_10m,precipitation',
      wind_speed_unit: 'kmh',
      timezone: 'auto'
    });
    fetch(`https://api.open-meteo.com/v1/forecast?${params}`, { signal: controller.signal })
      .then((response) => { if (!response.ok) throw new Error(`Weather request failed: ${response.status}`); return response.json(); })
      .then((payload) => {
        const current = payload.current;
        if (!current) throw new Error('Weather response did not include current conditions');
        setWeather({ loading: false, available: true, error: false, temperature: Number(current.temperature_2m), windSpeed: Number(current.wind_speed_10m), precipitation: Number(current.precipitation) });
      })
      .catch((error) => { if (error.name !== 'AbortError') setWeather({ loading: false, available: false, error: true }); });
    return () => controller.abort();
  }, [contextId, context.latitude, context.longitude]);
  useEffect(() => {
    if (contextId === 'none') {
      setApiPrediction(null);
      return undefined;
    }
    if (weather.loading) return undefined;
    const controller = new AbortController();
    const payload = {
      location_name: context.area,
      latitude: Number(context.latitude),
      longitude: Number(context.longitude),
      damage_assets: damage,
      crews_available: crews,
      access_percent: access,
      asset_type: context.device,
      circuit: context.circuit,
      event: stormProfiles[storm].label,
      wind_speed_kmh: weather.available ? weather.windSpeed : null,
      precipitation_mm: weather.available ? weather.precipitation : null,
      matched_utility_data: context.dataMode === 'matched',
      circuit_customer_count: context.customers,
      circuit_recloser_count: context.reclosers || 0,
      circuit_fuse_count: context.fuses || 0,
      circuit_switch_count: context.switches || 0,
      circuit_transformer_count: context.transformers || 0,
      topology_depth: context.topologyDepth || 0
    };
    fetch(`${apiBaseUrl}/predict`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload), signal: controller.signal })
      .then((response) => { if (!response.ok) throw new Error(`Prediction API failed: ${response.status}`); return response.json(); })
      .then((prediction) => setApiPrediction({ risk: prediction.risk.score, low: prediction.restoration_window_hours.low, high: prediction.restoration_window_hours.high, confidence: prediction.risk.confidence || prediction.confidence, dataMode: prediction.data_mode, dataSources: prediction.data_sources }))
      .catch((error) => { if (error.name !== 'AbortError') setApiPrediction(null); });
    return () => controller.abort();
  }, [contextId, context.latitude, context.longitude, context.device, context.circuit, context.dataMode, storm, damage, crews, access, weather]);

  const searchLocation = async () => {
    const query = locationQuery.trim();
    if (!query) return;
    const locationParts = query.split(',').map((part) => part.trim()).filter(Boolean);
    if (locationParts.length < 2) {
      setLocationStatus('Enter a city and state, for example Columbus, Ohio.');
      return;
    }
    setLocationStatus('Searching…');
    try {
      const params = new URLSearchParams({ q: query, format: 'jsonv2', addressdetails: '1', limit: '1' });
      const response = await fetch(`https://nominatim.openstreetmap.org/search?${params}`);
      if (!response.ok) throw new Error(`Location search failed: ${response.status}`);
      const results = await response.json();
      if (!results.length) throw new Error('No location found');
      const match = results[0];
      const address = match.address || {};
      const name = address.city || address.town || address.village || address.municipality;
      if (!name || !address.state) throw new Error('Search result is not a city and state');
      setLatitude(Number(match.lat));
      setLongitude(Number(match.lon));
      setLocationName(name);
      setContextId('custom');
      setLocationStatus(`${name} found`);
    } catch {
      setLocationStatus('Location not found. Try a city and state or country.');
    }
  };

  const showWelcomeScreen = () => {
    setWelcomeLeaving(false);
    setShowWelcome(true);
  };

  if (showWelcome) {
    return <WelcomeScreen leaving={welcomeLeaving} onNext={() => {
      setWelcomeLeaving(true);
      window.setTimeout(() => setShowWelcome(false), 420);
    }} />;
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="topbar-inner">
          <div className="brand"><img className="sparky-brand-avatar" src={sparkyFriendlyUrl} alt="Sparky" /><div><strong>PowerSpark</strong><span>AI outage guidance</span></div></div>
          <div className="topbar-meta"><button className="topbar-action" type="button" onClick={showWelcomeScreen}>Back to welcome</button><span className="assistant-status"><span className="live-dot" />Assistant online</span><span className="divider" /><span>Weather-aware guidance</span></div>
        </div>
      </header>
      <main><AssistantWorkspace context={context} contextId={contextId} setContextId={setContextId} locationQuery={locationQuery} setLocationQuery={setLocationQuery} searchLocation={searchLocation} locationStatus={locationStatus} latitude={latitude} longitude={longitude} setLatitude={setLatitude} setLongitude={setLongitude} storm={storm} setStorm={setStorm} result={result} riskLabel={riskLabel} weather={weather} damage={damage} setDamage={setDamage} crews={crews} setCrews={setCrews} access={access} setAccess={setAccess} priority={priority} setPriority={setPriority} activeTab={activeTab} setActiveTab={setActiveTab} apiPrediction={apiPrediction} /></main>
      <footer><span>◈ Estimates are informational and may change as field conditions are confirmed.</span><span>Challenge prototype · Privacy-first by design</span></footer>
    </div>
  );
}

function WelcomeScreen({ leaving, onNext }) {
  return <main className={`welcome-screen${leaving ? ' leaving' : ''}`}>
    <div className="welcome-glow" />
    <div className="welcome-content">
      <div className="welcome-brand"><img className="sparky-brand-avatar" src={sparkyFriendlyUrl} alt="Sparky" /><span>POWERSPARK</span></div>
      <div className="welcome-copy">
        <p className="eyebrow">AI-POWERED OUTAGE GUIDANCE</p>
        <h1>Clear answers when the power is at risk.</h1>
        <p>Understand outage risk, restoration timing, current weather, and the next steps to stay prepared.</p>
        <button className="welcome-next" type="button" onClick={onNext}>Explore the estimate <span>→</span></button>
      </div>
      <div className="welcome-foot"><span><i className="live-dot" />Live weather-aware guidance</span><span>Challenge prototype · Privacy-first by design</span></div>
    </div>
  </main>;
}

function AssistantWorkspace({ context, contextId, setContextId, locationQuery, setLocationQuery, searchLocation, locationStatus, latitude, longitude, setLatitude, setLongitude, storm, setStorm, result, riskLabel, weather, damage, setDamage, crews, setCrews, access, setAccess, priority, setPriority, activeTab, setActiveTab, apiPrediction }) {
  const [question, setQuestion] = useState('');
  const [messages, setMessages] = useState([{ id: 'welcome-message', role: 'assistant', text: 'Hi, I’m Sparky! I can explain outage risk, restoration timing, weather, preparation steps, and what data supports this estimate.' }]);
  const [chatMode, setChatMode] = useState('fallback');
  const [isAsking, setIsAsking] = useState(false);
  const chatHistoryRef = useRef(null);
  useEffect(() => {
    const history = chatHistoryRef.current;
    if (history) history.scrollTop = history.scrollHeight;
  }, [messages]);
  const ask = async (prompt) => {
    const text = prompt || question.trim();
    if (!text) return;
    if (context.dataMode === 'unselected') return;
    if (isAsking) return;
    const messageId = Date.now();
    const responseStartedAt = Date.now();
    const minimumResponsePause = 1100;
    setMessages((items) => [...items, { id: `${messageId}-user`, role: 'user', text }, { id: `${messageId}-pending`, role: 'assistant', text: 'Checking the live estimate…' }]);
    setQuestion('');
    setIsAsking(true);
    const payload = { question: text, prediction: { location_name: context.area, latitude: Number(context.latitude), longitude: Number(context.longitude), damage_assets: damage, crews_available: crews, access_percent: access, asset_type: context.device, circuit: context.circuit, event: stormProfiles[storm].label, wind_speed_kmh: weather.available ? weather.windSpeed : null, precipitation_mm: weather.available ? weather.precipitation : null, matched_utility_data: context.dataMode === 'matched', circuit_customer_count: context.customers, circuit_recloser_count: context.reclosers || 0, circuit_fuse_count: context.fuses || 0, circuit_switch_count: context.switches || 0, circuit_transformer_count: context.transformers || 0, topology_depth: context.topologyDepth || 0 } };
    try {
      const response = await fetch(`${apiBaseUrl}/chat`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      if (!response.ok) throw new Error(`Chat API failed: ${response.status}`);
      const data = await response.json();
      await new Promise((resolve) => window.setTimeout(resolve, Math.max(0, minimumResponsePause - (Date.now() - responseStartedAt))));
      setChatMode(data.mode);
      setMessages((items) => [...items.slice(0, -1), { id: `${messageId}-response`, role: 'assistant', text: data.answer }]);
    } catch {
      await new Promise((resolve) => window.setTimeout(resolve, Math.max(0, minimumResponsePause - (Date.now() - responseStartedAt))));
      setChatMode('fallback');
      setMessages((items) => [...items.slice(0, -1), { id: `${messageId}-response`, role: 'assistant', text: answerFor(text, result, context, weather, apiPrediction, storm, damage, crews, access) }]);
    } finally {
      setIsAsking(false);
    }
  };
  return <section className="assistant-shell">
    <section className="context-bar"><label className="location-search">City, state<input list="location-suggestions" value={locationQuery} onChange={(event) => { setLocationQuery(event.target.value); setLocationStatus(''); }} onKeyDown={(event) => { if (event.key === 'Enter') { event.preventDefault(); searchLocation(); } }} placeholder="Enter a city, state" aria-label="City and state" required /><datalist id="location-suggestions"><option value="Columbus, Ohio" /></datalist><button type="button" onClick={searchLocation}>Search</button>{locationStatus && <small>{locationStatus}</small>}</label><label>Storm outlook<select value={storm} onChange={(event) => setStorm(event.target.value)}>{Object.entries(stormProfiles).map(([key, item]) => <option key={key} value={key}>{item.label}</option>)}</select></label><div className="context-chip">⌖ <strong>{context.area}</strong><span>{context.dataMode === 'matched' ? 'Using local utility evidence' : context.dataMode === 'unselected' ? 'Enter a city and state to begin' : 'Using regional estimates'}</span></div></section>
    {contextId === 'custom' && <div className="coordinate-inputs compact"><label>Latitude<input type="number" step="0.001" value={latitude} onChange={(event) => setLatitude(event.target.value)} /></label><label>Longitude<input type="number" step="0.001" value={longitude} onChange={(event) => setLongitude(event.target.value)} /></label></div>}
    <div className="assistant-layout"><section className="chat-panel"><div className="chat-history" ref={chatHistoryRef}>{messages.map((message) => <div className={`chat-message ${message.role}`} key={message.id}>{message.role === 'assistant' ? <img className="sparky-avatar" src={sparkyAvatar(message.text, message.id)} alt="Sparky" /> : <span>You</span>}<p>{message.text}</p></div>)}</div><div className="quick-prompts">{['What is my outage risk?', 'When could power return?', 'How should I prepare?', 'What data supports this?'].map((prompt) => <button key={prompt} disabled={context.dataMode === 'unselected' || isAsking} onClick={() => ask(prompt)}>{prompt}</button>)}</div><form className="composer" onSubmit={(event) => { event.preventDefault(); ask(); }}><input value={question} disabled={context.dataMode === 'unselected' || isAsking} onChange={(event) => setQuestion(event.target.value)} placeholder={context.dataMode === 'unselected' ? 'Enter a city and state first…' : isAsking ? 'Sparky is checking the estimate…' : 'Ask about your outage estimate…'} aria-label="Ask the outage assistant" /><button type="submit" disabled={context.dataMode === 'unselected' || isAsking}>Ask <span>↗</span></button></form></section><section className="response-column"><ResponseCards result={result} riskLabel={riskLabel} context={context} storm={storm} weather={weather} apiPrediction={apiPrediction} setActiveTab={setActiveTab} /></section></div>
  </section>;
}

function answerFor(question, result, context, weather, apiPrediction, storm, damage, crews, access) {
  const lower = question.toLowerCase();
  if (lower.includes('prepare')) {
    const guidance = preparationGuidance(storm, result.risk);
    return `${guidance.title}. ${guidance.items.slice(0, 3).join(' ')} Follow official safety updates.`;
  }
  if (lower.includes('data') || lower.includes('support')) {
    const sources = apiPrediction?.dataSources || {};
    const network = sources.utility_network || 'generic network fallback';
    const history = sources.historical_damage || 'unavailable for this location';
    const weatherSource = sources.weather || (weather.available ? 'live request weather' : 'storm assumptions');
    return `For ${context.area}, the estimate uses ${network}; historical damage source: ${history}; weather source: ${weatherSource}. It also includes ${damage} damaged assets, ${crews} available crews, ${access}% access, and the ${stormProfiles[storm].label.toLowerCase()} outlook. The model estimates asset repair duration, not a confirmed customer outage.`;
  }
  if (lower.includes('when') || lower.includes('return') || lower.includes('restore')) return `The current estimated restoration window is ${result.low}–${result.high} hours if an outage occurs. This is an estimate, not a confirmed restoration time.`;
  return `The current outage risk is ${result.risk}% (${result.risk >= 70 ? 'high' : result.risk >= 45 ? 'moderate' : 'low'}). Confidence is ${result.confidence}%, and conditions can change as field reports arrive.`;
}

function ResponseCards({ result, riskLabel, context, storm, weather, apiPrediction, setActiveTab }) {
  const locationSelected = context.dataMode !== 'unselected';
  const guidance = preparationGuidance(context.dataMode === 'unselected' ? 'moderate' : storm, result.risk);
  const evidenceLabel = apiPrediction?.dataMode === 'regional_match' ? 'Nearby regional utility evidence' : apiPrediction?.dataMode === 'matched' ? 'Matched utility evidence' : locationSelected ? 'Generic regional estimate' : context.source;
  const evidenceDescription = apiPrediction?.dataSources?.historical_damage === 'nearest coordinate lookup' ? 'Historical damage and network patterns were found near this location.' : locationSelected ? 'No supplied utility records were found near this location.' : 'Enter a city and state to begin.';
  return <div className="response-cards"><article className={`answer-card risk-card ${locationSelected ? riskLabel.toLowerCase() : ''}`}><div className="card-kicker">OUTAGE RISK {locationSelected && <span>{result.confidence}% confidence</span>}</div><strong className="card-value">{locationSelected ? `${result.risk}%` : '—'}</strong><h3>{locationSelected ? `${riskLabel} likelihood` : 'Select a location'}</h3><p>{locationSelected ? (apiPrediction?.dataMode === 'regional_match' ? 'Nearby utility evidence is available.' : 'Generic estimates are being used.') : 'Enter a city and state to calculate outage risk.'}</p></article><article className="answer-card restoration-card"><div className="card-kicker">RESTORATION WINDOW</div><strong className="card-value">{locationSelected ? <>{result.low}–{result.high}<small> hrs</small></> : '—'}</strong><h3>{locationSelected ? 'If an outage occurs' : 'Awaiting location'}</h3><p>{locationSelected ? 'Storm-calibrated asset-repair estimate.' : 'Enter a city and state to estimate restoration timing.'}</p></article><article className="answer-card weather-card"><div className="card-kicker">WEATHER NOW</div><strong className="card-value weather-value">{weather.available ? `${Math.round(weather.windSpeed * 0.621371)} mph` : '—'}</strong><h3>{weather.available ? `${Math.round(weather.temperature * 9 / 5 + 32)}°F · ${(weather.precipitation / 25.4).toFixed(2)} in rain` : 'Weather unavailable'}</h3><p>{weather.available ? 'Live conditions are included in the estimate.' : 'Search for a city and state to load local weather.'}</p></article><article className="answer-card evidence-card"><div className="card-kicker">EVIDENCE</div><h3>{evidenceLabel}</h3><p>{evidenceDescription}</p></article><article className="answer-card actions-card"><div className="card-kicker">PREPARE</div>{locationSelected ? <><h3>{guidance.title}</h3><ul>{guidance.items.map((item) => <li key={item}>{item}</li>)}</ul></> : <><h3>Select a location first</h3><p>Enter a city and state to receive preparation steps based on the storm outlook and estimated outage risk.</p></>}</article><article className={`answer-card alert-card ${locationSelected && result.eligible ? 'alert-critical' : 'alert-monitor'}`}><div className="card-kicker">ALERTS</div><h3>{locationSelected ? (result.eligible ? 'Proactive alert recommended' : 'Continue monitoring') : 'Awaiting location'}</h3><p>{locationSelected ? (result.eligible ? 'Risk or declared need meets the current alert threshold.' : 'No alert threshold is met yet.') : 'Select a location to evaluate alert guidance.'}</p></article></div>;
}

function Overview({ result, riskLabel, damage, setDamage, crews, setCrews, access, setAccess }) {
  return <section className="dashboard-grid">
    <article className="panel risk-panel"><div className="panel-title"><div><p className="eyebrow">OUTAGE RISK</p><h2>{riskLabel} likelihood</h2></div><span className={`badge ${riskLabel.toLowerCase()}`}>{result.confidence}% confidence</span></div><div className="risk-score"><div className={`ring ${riskLabel.toLowerCase()}`}><strong>{result.risk}%</strong><span>estimated risk</span></div><div className="risk-copy"><strong>{result.risk >= 55 ? 'Plan for an interruption.' : 'An interruption is possible.'}</strong><p>This is a scenario estimate, not a confirmed outage. Risk can move as the storm track, damage reports, and crew availability change.</p></div></div><div className="factor-list"><Factor label="Storm exposure" value={result.risk > 65 ? 'Elevated' : 'Typical'} width={result.risk > 65 ? 82 : 55} /><Factor label="Network position" value="Downstream device" width={68} /><Factor label="Historical damage pattern" value="Tree contact likely" width={74} /></div></article>
    <article className="panel restore-panel"><div className="panel-title"><div><p className="eyebrow">RESTORATION WINDOW</p><h2>{result.low}–{result.high} hours</h2></div><span className="clock">◷</span></div><p className="muted">If an outage occurs, the current model expects restoration in this range.</p><div className="range-track"><span style={getRangeStyle(result.low, damage)} /></div><div className="range-labels"><span>First crews dispatched</span><span>Complex repairs</span></div><div className="confirmed"><span>✓</span><div><strong>What is confirmed</strong><p>No active outage is confirmed for this location.</p></div></div><div className="estimate"><span>≈</span><div><strong>What is estimated</strong><p>Duration reflects damage, crew, and access assumptions below.</p></div></div></article>
    <ScenarioPanel result={result} damage={damage} setDamage={setDamage} crews={crews} setCrews={setCrews} access={access} setAccess={setAccess} />
    <article className="panel message-panel"><div className="panel-title"><div><p className="eyebrow">PROACTIVE MESSAGE</p><h2>Ready to send</h2></div><span className="message-icon">✉</span></div><div className="message-preview">“A severe storm may affect your area. We estimate a <strong>{result.low}–{result.high} hour</strong> restoration window if an outage occurs. Charge devices, secure outdoor items, and keep emergency supplies nearby.”</div><button className="outline-button" onClick={() => navigator.clipboard?.writeText(`A severe storm may affect your area. Estimated restoration: ${result.low}-${result.high} hours if an outage occurs.`)}>Copy customer message <span>↗</span></button></article>
  </section>;
}

function Factor({ label, value, width }) { return <div className="factor"><div><span>{label}</span><strong>{value}</strong></div><div className="factor-bar"><i style={{ width: `${width}%` }} /></div></div>; }
function getRangeStyle(low, damage) {
  const left = Math.min(82, low * 5);
  const requestedWidth = Math.min(45, 20 + damage * 2);
  return { left: `${left}%`, width: `${Math.min(requestedWidth, 100 - left)}%` };
}
function ScenarioPanel({ result, damage, setDamage, crews, setCrews, access, setAccess }) { return <article className="panel scenario-panel"><div className="panel-title"><div><p className="eyebrow">MODEL ASSUMPTIONS</p><h2>Test the response</h2></div><span className="edit">↗ Live model</span></div><p className="muted">Move an input to see how the estimate responds.</p><Slider label="Likely damaged assets" value={damage} min={0} max={10} suffix=" assets" onChange={setDamage} /><Slider label="Crews available" value={crews} min={1} max={8} suffix="" onChange={setCrews} /><Slider label="Road access" value={access} min={20} max={100} suffix="%" onChange={setAccess} /><div className="assumption-foot"><span>Current window</span><strong>{result.low}–{result.high} hrs</strong></div></article>; }
function Slider({ label, value, min, max, suffix, onChange }) { return <label className="slider-row"><span>{label}<b>{value}{suffix}</b></span><input type="range" min={min} max={max} value={value} onChange={(event) => onChange(Number(event.target.value))} /></label>; }
function Explain({ result, context, storm, weather }) { return <section className="single-panel panel"><p className="eyebrow">EXPLAINABILITY</p><h2>What is driving this estimate?</h2><p className="intro">The concierge combines challenge-provided topology, asset, damage, labor references, and live weather when available. It does not expose raw facility or customer records.</p><div className="explain-grid"><div className="explain-card"><span className="number">01</span><h3>Storm conditions</h3><p><strong>{stormProfiles[storm].label}</strong> increases the chance of tree contact and access delays. Weather signal: <strong>{weatherDescription(weather)}</strong>.</p></div><div className="explain-card"><span className="number">02</span><h3>Network position</h3><p><strong>{context.circuit}</strong> is downstream of a protective device, so one fault can affect a local group of customers.</p></div><div className="explain-card"><span className="number">03</span><h3>Uncertainty</h3><p>Confidence is <strong>{result.confidence}%</strong>. The range widens when utility, weather, field damage, or access conditions are unknown.</p></div></div></section>; }
function Prepare({ result, priority, setPriority }) { return <section className="single-panel panel"><p className="eyebrow">PREPARE & STAY SAFE</p><h2>Actions matched to this scenario</h2><p className="intro">These are practical steps, not a substitute for emergency guidance. Never approach a downed line—assume it is energized and call 911 for immediate danger.</p><div className="prep-list"><div><span className="check">✓</span><p><strong>Charge essentials now</strong><small>Keep phones, medical devices, and backup batteries ready before conditions worsen.</small></p></div><div><span className="check">✓</span><p><strong>Secure outdoor items</strong><small>Bring in loose furniture and clear a safe path to your meter or entrance.</small></p></div><div><span className="check">!</span><p><strong>Plan for {result.high}+ hours without power</strong><small>Keep water, shelf-stable food, flashlights, and a manual can opener available.</small></p></div></div><label className="priority-toggle"><input type="checkbox" checked={priority} onChange={(event) => setPriority(event.target.checked)} /><span>Show guidance for a medical or accessibility need</span></label></section>; }
function Alerts({ result, context }) { return <section className="single-panel panel"><p className="eyebrow">ALERT ELIGIBILITY</p><div className="alert-header"><div><h2>{result.eligible ? 'Recommend proactive alert' : 'Monitor, no alert yet'}</h2><p className="intro">{result.eligible ? `Risk or declared need meets the challenge threshold for ${context.id}.` : 'The current estimate does not meet the risk threshold. Continue monitoring for a storm or damage update.'}</p></div><span className={`alert-status ${result.eligible ? 'recommended' : 'watch'}`}>{result.eligible ? 'Recommended' : 'Watch'}</span></div><div className="alert-cards"><div><span>01</span><strong>When</strong><p>Before storm arrival, then when an outage is confirmed or the estimate changes materially.</p></div><div><span>02</span><strong>Channel</strong><p>SMS or email selected by the customer—never reveal circuit or facility details.</p></div><div><span>03</span><strong>Escalate when</strong><p>Risk exceeds 70%, restoration exceeds 24 hours, or a safety condition is reported.</p></div></div></section>; }

const mapStyles = { primary: ['Primary lines', '#3f91b2'], secondary: ['Secondary lines', '#9b7770'], damage: ['Reported damage', '#6fb9d6'], transformer: ['Transformers', '#6d6d6d'], fuse: ['Fuse banks', '#b47722'], recloser: ['Reclosers', '#7c4a91'], switch: ['Switches', '#387b68'], customer: ['Customers', '#b6b6b6'] };
function MapPanel({ data, context }) {
  const [visible, setVisible] = useState({ primary: true, secondary: true, damage: true, transformer: true, fuse: false, recloser: true, switch: false, customer: false });
  const mapRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const layersRef = useRef({});
  if (!data) return <section className="single-panel panel map-loading"><p className="eyebrow">NETWORK MAP</p><h2>Loading supplied network data…</h2></section>;
  const toggle = (kind) => setVisible((current) => ({ ...current, [kind]: !current[kind] }));
  const shown = data.features.filter((feature) => visible[feature.kind]);
  useEffect(() => {
    if (!mapRef.current || !data) return undefined;
    const map = L.map(mapRef.current, { zoomControl: true, preferCanvas: true });
    mapInstanceRef.current = map;
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { attribution: '&copy; OpenStreetMap contributors', maxZoom: 19 }).addTo(map);
    const bounds = [];
    const colorFor = (kind) => mapStyles[kind][1];
    data.features.forEach((feature) => {
      const group = layersRef.current[feature.kind] ||= L.layerGroup();
      const options = { color: colorFor(feature.kind), weight: feature.kind === 'primary' ? 3 : 1.5, opacity: .85 };
      const layer = feature.geometry === 'LineString'
        ? L.polyline(feature.coordinates.map(([lng, lat]) => [lat, lng]), options)
        : L.circleMarker([feature.coordinates[1], feature.coordinates[0]], { ...options, radius: feature.kind === 'damage' ? 5 : 3, fillColor: colorFor(feature.kind), fillOpacity: .85 });
      layer.bindPopup(`<strong>${mapStyles[feature.kind][0]}</strong><br>${feature.id}<br>${feature.circuit ? `Circuit ${feature.circuit}` : 'Network asset'}`);
      layer.addTo(group);
      if (feature.geometry === 'LineString') feature.coordinates.forEach(([lng, lat]) => bounds.push([lat, lng]));
      else bounds.push([feature.coordinates[1], feature.coordinates[0]]);
    });
    Object.entries(layersRef.current).forEach(([kind, group]) => { if (visible[kind]) group.addTo(map); });
    if (bounds.length) map.fitBounds(bounds, { padding: [18, 18] });
    setTimeout(() => map.invalidateSize(), 0);
    return () => { map.remove(); mapInstanceRef.current = null; layersRef.current = {}; };
  }, [data]);
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;
    Object.entries(layersRef.current).forEach(([kind, group]) => { if (visible[kind]) group.addTo(map); else map.removeLayer(group); });
  }, [visible]);
  const selectedTopology = data.topology[0];
  return <section className="map-layout"><article className="panel map-panel"><div className="panel-title"><div><p className="eyebrow">NETWORK MAP</p><h2>Storm impact context</h2></div><span className="map-count">{shown.length} objects shown</span></div><p className="muted">Supplied GIS overlays on an OpenStreetMap basemap. Customer records are represented as anonymous points.</p><div className="map-canvas" ref={(node) => { mapRef.current = node; }} aria-label="Generalized electric distribution network map" /><div className="map-legend">{Object.entries(mapStyles).map(([kind, [label, color]]) => <button aria-pressed={visible[kind]} className={visible[kind] ? 'on' : ''} onClick={() => toggle(kind)} key={kind}><i style={{ background: color }} />{label}<b>{visible[kind] ? 'On' : 'Off'}</b></button>)}</div></article><aside className="panel topology-panel"><p className="eyebrow">UPSTREAM TOPOLOGY</p><h2>{context.circuit} path</h2><p className="muted">Representative relationships from <code>matched_upstream.txt</code>.</p><div className="topology-path"><span className="topology-node">Anonymous premise</span><b>↓</b><span className="topology-node">Distribution transformer</span><b>↓</b><span className="topology-node">Protective device</span><b>↓</b><span className="topology-node">Circuit breaker</span></div><div className="topology-detail"><strong>Data-linked example</strong><p>{selectedTopology?.upstream}</p><small>Anonymous premise {selectedTopology?.premise} · circuit {selectedTopology?.circuit}</small></div></aside></section>;
}

createRoot(document.getElementById('root')).render(<App />);
