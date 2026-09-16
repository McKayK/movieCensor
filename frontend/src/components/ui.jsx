import { thumbUrl } from '../api'

const STATUS_STYLES = {
  queued: ['Queued', 'bg-slate-700 text-slate-200'],
  analyzing: ['Analyzing', 'bg-sky-500/20 text-sky-300'],
  needs_review: ['Needs review', 'bg-amber-500/20 text-amber-300'],
  approved: ['Queued to render', 'bg-sky-500/20 text-sky-300'],
  rendering: ['Rendering', 'bg-sky-500/20 text-sky-300'],
  publishing: ['Publishing', 'bg-sky-500/20 text-sky-300'],
  done: ['Done', 'bg-emerald-500/20 text-emerald-300'],
  failed: ['Failed', 'bg-rose-500/20 text-rose-300'],
  canceled: ['Canceled', 'bg-slate-700 text-slate-400'],
  running: ['Scanning', 'bg-sky-500/20 text-sky-300'],
  confirmed: ['Confirmed', 'bg-emerald-500/20 text-emerald-300'],
  unconfirmed: ['Unconfirmed', 'bg-amber-500/20 text-amber-300'],
  audio_only: ['Heard in audio', 'bg-violet-500/20 text-violet-300'],
}

export function StatusBadge({ status }) {
  const [label, cls] = STATUS_STYLES[status] || [status, 'bg-slate-700 text-slate-200']
  return <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${cls}`}>{label}</span>
}

export function ProgressBar({ value, active = true }) {
  const pct = Math.round(Math.max(0, Math.min(1, value || 0)) * 100)
  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-slate-800">
      <div
        className={`h-full rounded-full transition-all duration-500 ${active ? 'bg-amber-500' : 'bg-slate-600'}`}
        style={{ width: `${pct}%` }}
      />
    </div>
  )
}

export function Poster({ thumb, title, className = '' }) {
  const src = thumbUrl(thumb)
  return src ? (
    <img src={src} alt={title} loading="lazy" className={`aspect-[2/3] w-full rounded-lg bg-slate-800 object-cover ${className}`} />
  ) : (
    <div className={`flex aspect-[2/3] w-full items-center justify-center rounded-lg bg-slate-800 p-3 text-center text-sm text-slate-400 ${className}`}>
      {title}
    </div>
  )
}

export function ErrorBox({ error }) {
  if (!error) return null
  return (
    <div className="rounded-lg border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
      {String(error.message || error)}
    </div>
  )
}

export function Spinner({ className = '' }) {
  return <span className={`inline-block h-4 w-4 animate-spin rounded-full border-2 border-slate-500 border-t-amber-400 ${className}`} />
}
