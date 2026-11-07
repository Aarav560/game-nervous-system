// Mineflayer bridge: tier-3 reflexes for Minecraft + a localhost JSON-lines socket for Python.
// Python (tiers 1-2) sends {"cmd": ...}; the bridge streams state snapshots every 100 ms.
// Binds 127.0.0.1 only and requires the shared token from GNS_BRIDGE_TOKEN.
//
//   GNS_BRIDGE_TOKEN=... node bridge.js --mc-port 25599 --port 47100 --base 25,5,16
'use strict'
const net = require('net')
const mineflayer = require('mineflayer')
const { pathfinder, Movements, goals } = require('mineflayer-pathfinder')

function arg (name, def) {
  const i = process.argv.indexOf('--' + name)
  return i >= 0 ? process.argv[i + 1] : def
}
const MC_PORT = parseInt(arg('mc-port', '25599'))
const MC_HOST = arg('mc-host', '127.0.0.1')
const PORT = parseInt(arg('port', '47100'))
const VERSION = arg('version', '1.21.4')
const BASE = arg('base', '0,5,0').split(',').map(Number)
const TOKEN = process.env.GNS_BRIDGE_TOKEN || ''
const HOSTILE = new Set(['zombie', 'skeleton', 'spider', 'creeper', 'husk', 'drowned', 'witch', 'pillager', 'vindicator'])
if (!TOKEN) { console.error('GNS_BRIDGE_TOKEN not set'); process.exit(2) }

const t0 = Date.now()
const now = () => (Date.now() - t0) / 1000
let bot = null
let client = null
let events = []
let reflex = { name: 'none', status: 'none', id: 0 }
let autonomic = { eat_below_food: 14 }

function send (obj) { if (client) client.write(JSON.stringify(obj) + '\n') }
function ev (kind, data) { events.push({ t: now(), kind, ...data }) }

// ------------------------------------------------------------------ reflexes
let gen = 0
function stopAll () {
  gen++
  if (!bot) return
  try { bot.pathfinder.setGoal(null) } catch (e) {}
  try { bot.stopDigging() } catch (e) {}
  bot.clearControlStates()
}

function hostiles (radius) {
  return Object.values(bot.entities).filter(e => e !== bot.entity && e.position &&
    (HOSTILE.has(e.name) || (e.type === 'player' && e.username !== bot.username)) &&
    e.position.distanceTo(bot.entity.position) <= radius)
    .sort((a, b) => a.position.distanceTo(bot.entity.position) - b.position.distanceTo(bot.entity.position))
}

function logIds () {
  return Object.values(bot.registry.blocksByName).filter(b => b.name.endsWith('_log')).map(b => b.id)
}

async function gotoNear (pos, range, my) {
  bot.pathfinder.setGoal(new goals.GoalNear(pos.x, pos.y, pos.z, range))
  const deadline = Date.now() + 20000
  while (gen === my && Date.now() < deadline) {
    if (bot.entity.position.distanceTo(pos) <= range + 0.6) break
    if (!bot.pathfinder.isMoving() && bot.entity.position.distanceTo(pos) <= range + 1.5) break
    await bot.waitForTicks(2)
  }
  bot.pathfinder.setGoal(null)
}

async function collectDrops (my) {
  for (let i = 0; i < 6 && gen === my; i++) {
    const items = Object.values(bot.entities).filter(e => e.name === 'item' && e.position &&
      e.position.distanceTo(bot.entity.position) < 8)
    if (!items.length) return
    items.sort((a, b) => a.position.distanceTo(bot.entity.position) - b.position.distanceTo(bot.entity.position))
    await gotoNear(items[0].position, 0.5, my)
    await bot.waitForTicks(6)
  }
}

const REFLEXES = {
  async gather (p, my) {
    const ids = logIds()
    while (gen === my) {
      const block = bot.findBlock({ matching: ids, maxDistance: 64, useExtraInfo: b => true })
      if (!block) { reflex.status = 'failed'; return }
      await gotoNear(block.position, 2, my)
      if (gen !== my) return
      // dig the whole column reachable from here, bottom-up
      for (let dy = 0; dy < 6 && gen === my; dy++) {
        const b = bot.blockAt(block.position.offset(0, dy, 0))
        if (!b || !b.name.endsWith('_log')) { if (dy > 0) break; else continue }
        if (!bot.canDigBlock(b)) break
        try { await bot.dig(b, true) } catch (e) { ev('dig_error', { msg: e.message }); break }
        ev('dug', { block: b.name })
      }
      await collectDrops(my)
    }
  },
  async goto (p, my) {
    const pos = p.place === 'base' ? { x: BASE[0] + 0.5, y: BASE[1], z: BASE[2] + 0.5 } : p
    const v = new (require('vec3').Vec3)(pos.x, pos.y, pos.z)
    await gotoNear(v, p.range || 1.5, my)
    if (gen === my) reflex.status = bot.entity.position.distanceTo(v) <= (p.range || 1.5) + 1.5 ? 'succeeded' : 'failed'
  },
  async disengage (p, my) {
    while (gen === my) {
      const hs = hostiles(20)
      if (!hs.length) { reflex.status = 'succeeded'; bot.clearControlStates(); return }
      const away = bot.entity.position.minus(hs[0].position).normalize().scaled(12).plus(bot.entity.position)
      bot.pathfinder.setGoal(new goals.GoalNear(away.x, away.y, away.z, 1))
      await bot.waitForTicks(10)
    }
  },
  async engage (p, my) {
    while (gen === my) {
      const t = p.target ? Object.values(bot.entities).find(e => e.username === p.target) : hostiles(16)[0]
      if (!t || !t.isValid) { reflex.status = 'succeeded'; bot.clearControlStates(); return }
      await bot.lookAt(t.position.offset(0, t.height * 0.8, 0), true)
      const d = t.position.distanceTo(bot.entity.position)
      bot.setControlState('forward', d > 2.5)
      bot.setControlState('sprint', d > 2.5)
      if (d < 3.2) bot.attack(t)
      await bot.waitForTicks(12)
    }
  },
  async heal (p, my) {
    const food = bot.inventory.items().find(i => ['golden_apple', 'bread', 'cooked_beef', 'apple', 'baked_potato'].includes(i.name))
    if (!food) { reflex.status = 'failed'; return }
    try { await bot.equip(food, 'hand'); await bot.consume(); reflex.status = 'succeeded' } catch (e) { reflex.status = 'failed' }
  },
  async explore (p, my) {
    const ang = Math.random() * Math.PI * 2
    const v = bot.entity.position.offset(24 * Math.cos(ang), 0, 24 * Math.sin(ang))
    await gotoNear(v, 2, my)
    if (gen === my) reflex.status = 'succeeded'
  },
  async wait (p, my) { bot.clearControlStates() }
}

function startReflex (name, params) {
  stopAll()
  const my = gen
  reflex = { name, status: 'running', id: my }
  const fn = REFLEXES[name]
  if (!fn) { reflex.status = 'failed'; return }
  fn(params || {}, my).then(() => {
    if (gen === my && reflex.status === 'running' && name !== 'gather' && name !== 'wait') reflex.status = 'succeeded'
  }).catch(e => { if (gen === my) { reflex.status = 'failed'; ev('reflex_error', { name, msg: e.message }) } })
}

// ------------------------------------------------------------------ autonomic (never escalated)
let eating = false
async function autonomicTick () {
  if (!bot || eating || !bot.entity) return
  if (bot.food < autonomic.eat_below_food && hostiles(8).length === 0) {
    const food = bot.inventory.items().find(i => ['bread', 'cooked_beef', 'apple', 'baked_potato'].includes(i.name))
    if (food) {
      eating = true
      try { await bot.equip(food, 'hand'); await bot.consume(); ev('autonomic', { action: 'eat' }) } catch (e) {}
      eating = false
    }
  }
}

// ------------------------------------------------------------------ state
function snapshot () {
  const p = bot.entity.position
  const v = bot.entity.velocity
  const inv = {}
  for (const it of bot.inventory.items()) inv[it.name] = (inv[it.name] || 0) + it.count
  const ents = Object.values(bot.entities).filter(e => e !== bot.entity && e.position && e.name !== 'item' &&
    e.position.distanceTo(p) < 24).map(e => ({
    id: String(e.id), name: e.username || e.name || e.type || 'unknown', kind: e.type === 'player' ? 'player' : (e.name || e.type || 'unknown'),
    hostile: HOSTILE.has(e.name) || (e.type === 'player'), x: e.position.x, z: e.position.z,
    vx: e.velocity.x * 20, vz: e.velocity.z * 20, hp: e.health ?? null
  }))
  const log = bot.findBlock({ matching: logIds(), maxDistance: 48 })
  const out = {
    type: 'state', t: now(), x: p.x, y: p.y, z: p.z, vx: v.x * 20, vz: v.z * 20,
    health: bot.health, food: bot.food, inv, entities: ents,
    reflex: { name: reflex.name, status: reflex.status }, base: { x: BASE[0] + 0.5, y: BASE[1], z: BASE[2] + 0.5 },
    nearest_log: log ? log.position.distanceTo(p) : null,
    controls: Object.fromEntries(['forward', 'back', 'left', 'right', 'jump', 'sprint', 'sneak'].map(k => [k, bot.getControlState(k)])),
    events
  }
  events = []
  return out
}

// ------------------------------------------------------------------ socket
function handle (msg) {
  switch (msg.cmd) {
    case 'reflex': startReflex(msg.name, msg.params); break
    case 'release_all': stopAll(); reflex = { name: 'none', status: 'none', id: gen }; send({ type: 'ack', cmd: 'release_all', controls: snapshot().controls }); break
    case 'config': Object.assign(autonomic, msg.autonomic || {}); break
    case 'give': if (bot) bot.chat(`/give ${bot.username} ${msg.item} ${msg.count || 1}`); break
    case 'quit': stopAll(); send({ type: 'ack', cmd: 'quit' }); setTimeout(() => process.exit(0), 100); if (bot) bot.quit(); break
  }
}

const server = net.createServer(sock => {
  let authed = false
  let buf = ''
  sock.on('data', d => {
    buf += d.toString()
    let i
    while ((i = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, i); buf = buf.slice(i + 1)
      let msg
      try { msg = JSON.parse(line) } catch (e) { continue }
      if (!authed) {
        if (msg.cmd === 'hello' && msg.token === TOKEN) { authed = true; client = sock; send({ type: 'hello', ok: true }) } else { sock.end(); return }
        continue
      }
      handle(msg)
    }
  })
  sock.on('close', () => {
    if (client === sock) { client = null; stopAll() } // Python gone -> release everything
  })
})
server.listen(PORT, '127.0.0.1')

bot = mineflayer.createBot({ host: MC_HOST, port: MC_PORT, username: arg('username', 'gns_bot'), version: VERSION, auth: 'offline' })
bot.loadPlugin(pathfinder)
bot.on('error', e => ev('bot_error', { msg: e.message }))
bot.on('kicked', r => { ev('kicked', { reason: String(r) }); send({ type: 'kicked', reason: String(r) }) })
bot.on('end', () => { send({ type: 'end' }); setTimeout(() => process.exit(0), 200) })
bot.on('health', () => ev('health', { health: bot.health, food: bot.food }))
bot.on('playerCollect', (collector, item) => { if (collector === bot.entity) ev('collect', {}) })
bot.once('spawn', () => {
  const m = new Movements(bot)
  m.canDig = false
  m.allow1by1towers = false
  bot.pathfinder.setMovements(m)
  console.log(JSON.stringify({ ready: true, bridge_port: PORT }))
  setInterval(() => { if (client && bot.entity) send(snapshot()) }, 100)
  setInterval(autonomicTick, 500)
})
