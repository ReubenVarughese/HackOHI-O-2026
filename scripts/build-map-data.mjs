import fs from 'node:fs/promises';
import path from 'node:path';
import shp from 'shpjs';

const root = process.cwd();
const dataRoot = path.join(root, 'aep_challenge_hack26-main', 'data', 'raw');
const output = path.join(root, 'public', 'map-data.json');

function sample(features, limit) {
  if (features.length <= limit) return features;
  const step = features.length / limit;
  return Array.from({ length: limit }, (_, index) => features[Math.floor(index * step)]);
}

function normalizeFeature(feature, index, kind) {
  const coordinates = feature.geometry.coordinates;
  const properties = feature.properties || {};
  return {
    id: String(properties.id ?? properties.Outage_Num ?? `${kind}-${index}`),
    kind,
    circuit: String(properties.circuit_id ?? properties.Circuit ?? properties.aep_circui ?? ''),
    geometry: feature.geometry.type,
    coordinates
  };
}

const data1 = await shp(await fs.readFile(path.join(dataRoot, 'Data1.zip')));
const data2 = await shp(await fs.readFile(path.join(dataRoot, 'Data2.zip')));
const layers = [...data1, ...data2];
const getLayer = (name) => layers.find((layer) => layer.fileName === name);

const layerDefinitions = [
  ['Primary_ExportFeatures', 'primary', 140],
  ['Secondary_ExportFeatures', 'secondary', 180],
  ['arcos_prod_damage_09_28_Clip', 'damage', 180],
  ['Fuse_Bank_Clip', 'fuse', 55],
  ['Recloser_Bank_Clip', 'recloser', 35],
  ['Transformer_Clip', 'transformer', 100],
  ['Switch_Clip', 'switch', 45],
  ['Customer_Clip', 'customer', 120]
];

const features = layerDefinitions.flatMap(([source, kind, limit]) => {
  const layer = getLayer(source);
  return sample(layer?.features ?? [], limit).map((feature, index) => normalizeFeature(feature, index, kind));
});

const topologyArchive = await fs.readFile(path.join(dataRoot, 'matched_upstream.zip'));
const topologyText = await import('node:zlib').then(async ({ default: zlib }) => {
  const { execFile } = await import('node:child_process');
  return new Promise((resolve, reject) => {
    const temp = path.join(root, '.topology-temp.zip');
    fs.writeFile(temp, topologyArchive)
      .then(() => execFile('powershell.exe', ['-NoProfile', '-Command', `Expand-Archive -LiteralPath '${temp}' -DestinationPath '${temp}-dir' -Force; Get-Content '${temp}-dir\\matched_upstream.txt' -TotalCount 180`], (error, stdout) => {
        fs.rm(temp, { force: true }).catch(() => {});
        fs.rm(`${temp}-dir`, { recursive: true, force: true }).catch(() => {});
        if (error) reject(error); else resolve(stdout);
      }))
      .catch(reject);
  });
});

const topology = topologyText.trim().split(/\r?\n/).filter(Boolean).slice(0, 80).map((line) => {
  const parts = line.split('\t');
  return { premise: parts[0] || 'anonymous', upstream: parts.slice(1, 4).join(' → '), circuit: parts.at(-1)?.match(/\*([^*]+)$/)?.[1] || 'unknown' };
});

await fs.mkdir(path.dirname(output), { recursive: true });
await fs.writeFile(output, JSON.stringify({ generatedAt: new Date().toISOString(), features, topology }, null, 0));
console.log(`Wrote ${features.length} map features and ${topology.length} topology summaries to ${output}`);
