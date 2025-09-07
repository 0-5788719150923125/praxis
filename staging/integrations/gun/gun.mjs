import Gun from 'gun'
import SEA from 'gun/sea.js'

const bootstrapPeers = ['wss://59.src.eco/gun', 'wss://95.src.eco/gun']
const gun = Gun({
  peers: bootstrapPeers,
  file: `build/gun`,
  radisk: false,
  axe: false
})

const src = gun.get('src')
src.on((data) => {})

async function keepAlive() {
  const peers = gun.back('opt.peers')
  for (const i of bootstrapPeers) {
    const state = peers[i]?.wire?.readyState
    if (state === 0 || state === null || typeof state === 'undefined') {
      gun.opt({ peers: [...bootstrapPeers] })
    }
  }
  setTimeout(keepAlive, 15000)
}

keepAlive()

const cache = []
src
  .get('bullets')
  .get('trade')
  .on(async (data, key) => {
    try {
      const payload = JSON.parse(data)

      let message = payload.message
      if (payload?.pubKey !== null) {
        const sender = await gun.user(payload.pubKey)
        if (typeof sender !== 'undefined') {
          message = await SEA.verify(payload.message, sender.pub)
        }
      }

      if (!cache.includes(message)) {
        cache.push(message)
        process.stdout.write(message + '\n')
      }

      while (cache.length > 25) {
        cache.shift()
      }
    } catch (err) {}
  })
