import Gun from 'gun'

const bootstrapPeers = ['wss://59.src.eco/gun', 'wss://95.src.eco/gun']
const gun = Gun({
  peers: bootstrapPeers,
  file: `data/gun`,
  radisk: false,
  axe: false
})

const src = gun.get('src')
src.on((data) => {})

async function managePeers() {
  const peers = gun.back('opt.peers')
  for (const i of bootstrapPeers) {
    const state = peers[i]?.wire?.readyState
    if (state === 0 || state === null || typeof state === 'undefined') {
      gun.opt({ peers: [...bootstrapPeers] })
    }
  }
  setTimeout(managePeers, 15000)
}

managePeers()

const cache = []
src
  .get('bullets')
  .get('trade')
  .on(async (node, key) => {
    try {
      const payload = JSON.parse(node).message
      if (!cache.includes(payload)) {
        cache.push(payload)
        process.stdout.write(payload + '\n')
      }
      while (cache.length > 25) {
        cache.shift()
      }
    } catch (err) {}
  })
