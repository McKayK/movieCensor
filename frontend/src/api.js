export class ApiError extends Error {
  constructor(message, status) {
    super(message)
    this.status = status
  }
}

export async function api(path, { method = 'GET', body } = {}) {
  const res = await fetch(path, {
    method,
    credentials: 'same-origin',
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  })
  const data = await res.json().catch(() => ({}))
  if (!res.ok) {
    throw new ApiError(typeof data.detail === 'string' ? data.detail : res.statusText, res.status)
  }
  return data
}

export const thumbUrl = (thumb, w = 300, h = 450) =>
  thumb ? `/api/thumb?path=${encodeURIComponent(thumb)}&w=${w}&h=${h}` : null

export function fmtTime(sec) {
  if (sec == null) return '—'
  const h = Math.floor(sec / 3600)
  const m = Math.floor((sec % 3600) / 60)
  const s = (sec % 60).toFixed(1).padStart(4, '0')
  return h ? `${h}:${String(m).padStart(2, '0')}:${s}` : `${m}:${s}`
}

export function timeAgo(ts) {
  if (!ts) return ''
  const d = Date.now() / 1000 - ts
  if (d < 60) return 'just now'
  if (d < 3600) return `${Math.floor(d / 60)}m ago`
  if (d < 86400) return `${Math.floor(d / 3600)}h ago`
  return `${Math.floor(d / 86400)}d ago`
}

export const ACTIVE_STATUSES = ['queued', 'analyzing', 'needs_review', 'approved', 'rendering', 'publishing']
