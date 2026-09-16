import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { useUser } from '../App'
import { ACTIVE_STATUSES, api, fmtTime, timeAgo } from '../api'
import { ErrorBox, ProgressBar, Spinner, StatusBadge } from '../components/ui'
import { Line, RevealToggle, Word } from '../mask'

const DECISION_LABELS = { mute: 'Mute word', mute_line: 'Mute line', skip: 'Leave in' }

export default function Job() {
  const { id } = useParams()
  const { user } = useUser()
  const navigate = useNavigate()
  const [job, setJob] = useState(null)
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)
  const [filter, setFilter] = useState('all')
  const [playing, setPlaying] = useState(null)
  const audioRef = useRef(null)

  const load = useCallback(() => api(`/api/jobs/${id}`).then(setJob).catch(setError), [id])

  useEffect(() => {
    load()
  }, [load])

  useEffect(() => {
    if (!job || !ACTIVE_STATUSES.includes(job.status) || job.status === 'needs_review') return
    const t = setInterval(load, 2000)
    return () => clearInterval(t)
  }, [job?.status, load]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => () => audioRef.current?.pause(), [])

  const act = async (fn) => {
    setBusy(true)
    setError(null)
    try {
      await fn()
      await load()
    } catch (e) {
      setError(e)
    } finally {
      setBusy(false)
    }
  }

  const decide = (hit, decision) =>
    act(() => api(`/api/jobs/${id}/hits/${hit.id}`, { method: 'POST', body: { decision } }))

  const bulk = (status, decision) => act(() => api(`/api/jobs/${id}/hits`, { method: 'POST', body: { status, decision } }))

  const play = (hit, variant) => {
    const key = `${hit.id}-${variant}`
    audioRef.current?.pause()
    if (playing === key) {
      setPlaying(null)
      return
    }
    const a = new Audio(`/api/jobs/${id}/hits/${hit.id}/preview?variant=${variant}&t=${Date.now()}`)
    audioRef.current = a
    setPlaying(key)
    a.onended = () => setPlaying(null)
    a.onerror = () => {
      setPlaying(null)
      setError('Preview failed to load')
    }
    a.play().catch(() => setPlaying(null))
  }

  const hits = useMemo(() => {
    if (!job?.hits) return []
    if (filter === 'all') return job.hits
    return job.hits.filter((h) => h.status === filter)
  }, [job, filter])

  if (!job) {
    return error ? <ErrorBox error={error} /> : <Spinner />
  }

  const reviewing = job.status === 'needs_review'
  const running = ACTIVE_STATUSES.includes(job.status) && !reviewing
  const canPreview = job.hits?.length > 0 && !['failed'].includes(job.status)

  return (
    <div className="space-y-6">
      <div className="card space-y-4 p-5">
        <div className="flex flex-wrap items-center gap-3">
          <Link to={`/movie/${job.ratingKey}`} className="text-xl font-semibold hover:text-amber-300">
            {job.title} <span className="font-normal text-slate-500">{job.year}</span>
          </Link>
          <StatusBadge status={job.status} />
          <span className="ml-auto text-xs text-slate-500">
            Job #{job.id} · {job.username} · {timeAgo(job.createdAt)}
          </span>
        </div>
        {(running || job.status === 'done') && <ProgressBar value={job.status === 'done' ? 1 : job.progress} active={running} />}
        <div className="text-sm text-slate-300">{job.stage}</div>
        {job.error && <ErrorBox error={job.error} />}
        <ErrorBox error={error} />
        {job.outputPath && (
          <div className="rounded-lg bg-emerald-500/10 px-4 py-3 text-sm text-emerald-200">
            Saved to <span className="font-mono break-all">{job.outputPath}</span>. It shows up in the Censored library after Plex scans.
          </div>
        )}
        <div className="flex flex-wrap items-center gap-2 text-xs text-slate-400">
          <span>Words: {job.settings.groupLabels.join(', ') || '—'}</span>
          {job.settings.custom_words?.length > 0 && <span>· Extra: {job.settings.custom_words.length}</span>}
          <span>· Style: {job.settings.style}</span>
          {job.editCount > 0 && <span>· {job.editCount} edits</span>}
        </div>
        {job.canModify && (
          <div className="flex flex-wrap gap-2">
            {reviewing && (
              <button className="btn-primary" disabled={busy || job.pending > 0} onClick={() => act(() => api(`/api/jobs/${id}/approve`, { method: 'POST' }))}>
                {busy && <Spinner />} Approve & render
              </button>
            )}
            {ACTIVE_STATUSES.includes(job.status) && !job.cancelRequested && (
              <button className="btn-ghost" disabled={busy} onClick={() => act(() => api(`/api/jobs/${id}/cancel`, { method: 'POST' }))}>
                Cancel
              </button>
            )}
            {['failed', 'canceled'].includes(job.status) && (
              <button className="btn-ghost" disabled={busy} onClick={() => act(() => api(`/api/jobs/${id}/retry`, { method: 'POST' }))}>
                Retry
              </button>
            )}
            {user.isAdmin && !running && (
              <button
                className="btn-danger ml-auto"
                disabled={busy}
                onClick={async () => {
                  const withFile = job.outputPath ? window.confirm('Also delete the censored file from the Censored library?') : false
                  await act(() => api(`/api/jobs/${id}?delete_file=${withFile}`, { method: 'DELETE' }))
                  navigate('/jobs')
                }}
              >
                Delete job
              </button>
            )}
          </div>
        )}
        {reviewing && job.pending > 0 && (
          <p className="text-sm text-amber-300">
            {job.pending} word{job.pending === 1 ? ' is' : 's are'} in the subtitles but weren't heard clearly. Preview each one, then choose to mute
            the line or leave it in.
          </p>
        )}
      </div>

      {job.hits?.length > 0 && (
        <div className="space-y-3">
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex rounded-lg bg-slate-800 p-1 text-sm">
              {[
                ['all', `All ${job.hits.length}`],
                ['confirmed', `Confirmed ${job.counts.confirmed}`],
                ['unconfirmed', `Unconfirmed ${job.counts.unconfirmed}`],
                ['audio_only', `Audio only ${job.counts.audio_only}`],
              ].map(([v, label]) => (
                <button key={v} onClick={() => setFilter(v)} className={`rounded-md px-3 py-1 ${filter === v ? 'bg-slate-950 text-white' : 'text-slate-400'}`}>
                  {label}
                </button>
              ))}
            </div>
            {reviewing && job.canModify && job.counts.unconfirmed > 0 && (
              <div className="flex gap-2">
                <button className="btn-ghost py-1.5 text-xs" disabled={busy} onClick={() => bulk('unconfirmed', 'mute_line')}>
                  Mute all unconfirmed lines
                </button>
                <button className="btn-ghost py-1.5 text-xs" disabled={busy} onClick={() => bulk('unconfirmed', 'skip')}>
                  Leave all unconfirmed in
                </button>
              </div>
            )}
            <div className="ml-auto">
              <RevealToggle />
            </div>
          </div>

          <div className="card overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-left text-xs text-slate-500">
                <tr className="border-b border-slate-800">
                  <th className="px-4 py-2 font-medium">Time</th>
                  <th className="px-4 py-2 font-medium">Word</th>
                  <th className="px-4 py-2 font-medium">Line</th>
                  <th className="px-4 py-2 font-medium">Status</th>
                  <th className="px-4 py-2 font-medium">Preview</th>
                  <th className="px-4 py-2 font-medium">Decision</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {hits.map((h) => (
                  <tr key={h.id} className={h.decision === 'pending' ? 'bg-amber-500/5' : ''}>
                    <td className="px-4 py-2 font-mono text-xs whitespace-nowrap text-slate-400">{fmtTime(h.wordStart ?? h.cueStart)}</td>
                    <td className="px-4 py-2 whitespace-nowrap">
                      <Word text={h.word} />
                      <div className="text-xs text-slate-500">{h.label}</div>
                    </td>
                    <td className="max-w-md px-4 py-2 text-slate-300">
                      {h.line ? <Line line={h.line} cs={h.source !== 'audio' ? h.cs : null} ce={h.source !== 'audio' ? h.ce : null} word={h.word} /> : <span className="text-slate-600">—</span>}
                    </td>
                    <td className="px-4 py-2 whitespace-nowrap">
                      <StatusBadge status={h.status} />
                    </td>
                    <td className="px-4 py-2 whitespace-nowrap">
                      {canPreview && (
                        <div className="flex gap-1">
                          {['original', 'censored'].map((v) => (
                            <button
                              key={v}
                              onClick={() => play(h, v)}
                              className={`rounded px-2 py-1 text-xs ${playing === `${h.id}-${v}` ? 'bg-amber-500 text-slate-950' : 'bg-slate-800 text-slate-300 hover:bg-slate-700'}`}
                              title={v === 'original' ? 'Play the original audio' : 'Play with the current decision applied'}
                            >
                              {playing === `${h.id}-${v}` ? '■' : '▶'} {v === 'original' ? 'Orig' : 'Cens'}
                            </button>
                          ))}
                        </div>
                      )}
                    </td>
                    <td className="px-4 py-2 whitespace-nowrap">
                      {reviewing && job.canModify ? (
                        <div className="flex gap-1">
                          {Object.entries(DECISION_LABELS)
                            .filter(([d]) => (d === 'mute' ? h.muteStart != null : d === 'mute_line' ? h.cueStart != null : true))
                            .map(([d, label]) => (
                              <button
                                key={d}
                                disabled={busy}
                                onClick={() => decide(h, d)}
                                className={`rounded px-2 py-1 text-xs ${h.decision === d ? 'bg-amber-500 text-slate-950' : 'bg-slate-800 text-slate-300 hover:bg-slate-700'}`}
                              >
                                {label}
                              </button>
                            ))}
                        </div>
                      ) : (
                        <span className="text-xs text-slate-400">{DECISION_LABELS[h.decision] || 'Needs decision'}</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      {job.status === 'done' && job.hits?.length === 0 && (
        <p className="text-sm text-slate-400">Nothing was flagged. The copy only drops image subtitles and extra audio tracks.</p>
      )}
    </div>
  )
}
