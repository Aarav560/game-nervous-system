// Local test server for the Minecraft adapter: flying-squid (MIT), survival, offline mode,
// bound to 127.0.0.1 only. This is an `own_server` play context (Architecture §14).
// Plants a forest of oak-log columns around the base so "gather 64 logs" is repeatable.
//
//   node server.js --port 25599 --seed 1 --trees 24
'use strict'
const mcServer = require('flying-squid')

function arg (name, def) {
  const i = process.argv.indexOf('--' + name)
  return i >= 0 ? process.argv[i + 1] : def
}
const PORT = parseInt(arg('port', '25599'))
const SEED = parseInt(arg('seed', '1'))
const TREES = parseInt(arg('trees', '24'))
const VERSION = arg('version', '1.21.4')

let s = SEED
function rnd () { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff }

const serv = mcServer.createMCServer({
  motd: 'gns test server', port: PORT, host: '127.0.0.1', 'max-players': 4, 'online-mode': false,
  logging: false, gameMode: 0, difficulty: 0, worldFolder: undefined,
  generation: { name: 'superflat', options: { worldHeight: 80 } },
  kickTimeout: 10000, plugins: {}, modpe: false, 'view-distance': 6,
  'player-list-text': { header: { text: '' }, footer: { text: '' } },
  'everybody-op': false, 'max-entities': 200, version: VERSION
})

serv.on('listening', async () => {
  const spawn = serv.overworld.spawnPoint || { x: 0, y: 5, z: 0 }
  const logState = serv.registry.blocksByName.oak_log.defaultState
  const base = { x: Math.floor(spawn.x), y: Math.floor(spawn.y), z: Math.floor(spawn.z) }
  const placed = []
  let guard = 0
  while (placed.length < TREES && guard++ < 10000) {
    const ang = rnd() * Math.PI * 2
    const r = 7 + rnd() * 14
    const x = Math.floor(base.x + r * Math.cos(ang))
    const z = Math.floor(base.z + r * Math.sin(ang))
    if (placed.some(p => Math.abs(p.x - x) < 3 && Math.abs(p.z - z) < 3)) continue
    const h = 3 + Math.floor(rnd() * 2)
    for (let dy = 0; dy < h; dy++) {
      await serv.overworld.setBlockStateId(new (require('vec3').Vec3)(x, base.y + dy, z), logState)
    }
    placed.push({ x, z, h })
  }
  const logs = placed.reduce((a, p) => a + p.h, 0)
  // One machine-readable line for the Python launcher.
  console.log(JSON.stringify({ ready: true, port: PORT, version: VERSION, base, trees: placed.length, logs }))
})
