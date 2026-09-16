import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { ACTIVE_STATUSES, api } from '../api'
import { ErrorBox, Poster, Spinner } from '../components/ui'

export default function Library() {
  const [movies, setMovies] = useState(null)
  const [error, setError] = useState(null)
  const [q, setQ] = useState('')
  const [filter, setFilter] = useState('all')

  const load = (refresh = false) => {
    setError(null)
    api(`/api/movies${refresh ? '?refresh=true' : ''}`)
      .then(setMovies)
      .catch(setError)
  }
  useEffect(() => load(), [])

  const shown = useMemo(() => {
    if (!movies) return []
    const needle = q.trim().toLowerCase()
    return movies.filter((m) => {
      if (needle && !m.title.toLowerCase().includes(needle)) return false
      if (filter === 'censored') return m.jobStatus === 'done'
      if (filter === 'progress') return ACTIVE_STATUSES.includes(m.jobStatus)
      return true
    })
  }, [movies, q, filter])

  return (
    <div>
      <div className="mb-6 flex flex-wrap items-center gap-3">
        <input
          className="input max-w-md"
          placeholder="Search movies…"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          autoFocus
        />
        <select className="input w-auto" value={filter} onChange={(e) => setFilter(e.target.value)}>
          <option value="all">All movies</option>
          <option value="censored">Already censored</option>
          <option value="progress">In progress</option>
        </select>
        <button className="btn-ghost" onClick={() => load(true)}>
          Refresh from Plex
        </button>
        {movies && <span className="text-sm text-slate-500">{shown.length} of {movies.length}</span>}
      </div>

      <ErrorBox error={error} />
      {!movies && !error && (
        <div className="flex justify-center py-20">
          <Spinner className="h-6 w-6" />
        </div>
      )}

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6">
        {shown.map((m) => (
          <Link key={m.ratingKey} to={`/movie/${m.ratingKey}`} className="group relative">
            <Poster thumb={m.thumb} title={m.title} className="ring-amber-500/0 transition group-hover:ring-2 group-hover:ring-amber-500" />
            {m.jobStatus === 'done' && (
              <span className="absolute top-2 left-2 rounded bg-emerald-500 px-1.5 py-0.5 text-[10px] font-bold tracking-wide text-slate-950 uppercase">
                Censored
              </span>
            )}
            {ACTIVE_STATUSES.includes(m.jobStatus) && (
              <span className="absolute top-2 left-2 rounded bg-sky-500 px-1.5 py-0.5 text-[10px] font-bold tracking-wide text-slate-950 uppercase">
                In progress
              </span>
            )}
            <div className="mt-2 truncate text-sm font-medium">{m.title}</div>
            <div className="text-xs text-slate-500">
              {m.year} {m.contentRating && `· ${m.contentRating}`}
            </div>
          </Link>
        ))}
      </div>
    </div>
  )
}
