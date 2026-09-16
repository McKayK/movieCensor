import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { ACTIVE_STATUSES, api, timeAgo } from '../api'
import { ErrorBox, ProgressBar, Spinner, StatusBadge } from '../components/ui'

export default function Jobs() {
  const [jobs, setJobs] = useState(null)
  const [mine, setMine] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    let alive = true
    const load = () =>
      api(`/api/jobs${mine ? '?mine=true' : ''}`)
        .then((j) => alive && setJobs(j))
        .catch((e) => alive && setError(e))
    load()
    const t = setInterval(load, 3000)
    return () => {
      alive = false
      clearInterval(t)
    }
  }, [mine])

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <h1 className="text-xl font-semibold">Jobs</h1>
        <div className="ml-auto flex rounded-lg bg-slate-800 p-1 text-sm">
          {[
            [false, 'Everyone'],
            [true, 'Mine'],
          ].map(([v, label]) => (
            <button
              key={label}
              onClick={() => setMine(v)}
              className={`rounded-md px-3 py-1 ${mine === v ? 'bg-slate-950 text-white' : 'text-slate-400'}`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>
      <ErrorBox error={error} />
      {!jobs && !error && <Spinner />}
      {jobs?.length === 0 && <p className="text-sm text-slate-400">No jobs yet. Pick a movie in the Library to start one.</p>}
      <div className="card divide-y divide-slate-800">
        {jobs?.map((j) => (
          <Link key={j.id} to={`/jobs/${j.id}`} className="block px-4 py-3 hover:bg-slate-800/40">
            <div className="flex flex-wrap items-center gap-3">
              <span className="font-medium">
                {j.title} <span className="text-slate-500">{j.year}</span>
              </span>
              <StatusBadge status={j.status} />
              <span className="ml-auto text-xs text-slate-500">
                {j.username} · {timeAgo(j.createdAt)}
              </span>
            </div>
            <div className="mt-2 flex items-center gap-3">
              <div className="w-48 shrink-0">
                <ProgressBar value={j.status === 'done' ? 1 : j.progress} active={ACTIVE_STATUSES.includes(j.status)} />
              </div>
              <span className="truncate text-xs text-slate-400">{j.error || j.stage}</span>
            </div>
          </Link>
        ))}
      </div>
    </div>
  )
}
