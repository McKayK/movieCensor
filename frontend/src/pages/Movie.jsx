import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { ACTIVE_STATUSES, api, fmtTime, timeAgo } from '../api'
import { ErrorBox, Poster, Spinner, StatusBadge } from '../components/ui'
import { Line, RevealToggle, Word } from '../mask'

export default function Movie() {
  const { key } = useParams()
  const navigate = useNavigate()
  const [movie, setMovie] = useState(null)
  const [wordgroups, setWordgroups] = useState(null)
  const [scan, setScan] = useState(null)
  const [option, setOption] = useState('')
  const [error, setError] = useState(null)
  const [selected, setSelected] = useState(new Set())
  const [custom, setCustom] = useState('')
  const [style, setStyle] = useState('mute')
  const [echo, setEcho] = useState('deep')
  const [creating, setCreating] = useState(false)
  const [expanded, setExpanded] = useState(null)

  useEffect(() => {
    setMovie(null)
    setError(null)
    api(`/api/movies/${key}`)
      .then((m) => {
        setMovie(m)
        setScan(m.scan)
      })
      .catch(setError)
    api('/api/wordgroups').then(setWordgroups).catch(() => {})
  }, [key])

  // Poll while a scan is running.
  useEffect(() => {
    if (scan?.status !== 'running') return
    const t = setInterval(() => {
      api(`/api/scans/${scan.id}`).then(setScan).catch(setError)
    }, 1000)
    return () => clearInterval(t)
  }, [scan?.id, scan?.status])

  // Default selection: the Standard preset, once groups are known.
  useEffect(() => {
    if (wordgroups && selected.size === 0) {
      const std = wordgroups.presets.find((p) => p.id === 'standard')
      if (std) setSelected(new Set(std.groups))
    }
  }, [wordgroups]) // eslint-disable-line react-hooks/exhaustive-deps

  const startScan = async (force = false) => {
    setError(null)
    try {
      setScan(await api(`/api/movies/${key}/scan`, { method: 'POST', body: { option: option || null, force } }))
    } catch (e) {
      setError(e)
    }
  }

  const customWords = custom
    .split(',')
    .map((w) => w.trim())
    .filter(Boolean)

  const selectedHits = useMemo(() => (scan?.hits || []).filter((h) => selected.has(h.group)), [scan, selected])
  const activeJob = movie?.jobs?.find((j) => ACTIVE_STATUSES.includes(j.status))

  const createJob = async () => {
    setCreating(true)
    setError(null)
    try {
      const job = await api('/api/jobs', {
        method: 'POST',
        body: { ratingKey: key, scanId: scan.id, groups: [...selected], customWords, style, echo },
      })
      navigate(`/jobs/${job.id}`)
    } catch (e) {
      setError(e)
      setCreating(false)
    }
  }

  const toggle = (id) => {
    const next = new Set(selected)
    next.has(id) ? next.delete(id) : next.add(id)
    setSelected(next)
  }

  if (!movie) {
    return error ? (
      <ErrorBox error={error} />
    ) : (
      <div className="flex justify-center py-20">
        <Spinner className="h-6 w-6" />
      </div>
    )
  }

  const audio = movie.audio?.find((a) => a.index === movie.chosenAudio)
  const textOptions = (movie.subtitleOptions || []).filter((o) => o.text)
  const groupsWithHits = scan?.groups?.filter((g) => g.count > 0) || []
  const groupsWithout = scan?.groups?.filter((g) => g.count === 0) || []

  return (
    <div className="grid gap-8 lg:grid-cols-[240px_1fr]">
      <aside className="space-y-4">
        <Poster thumb={movie.thumb} title={movie.title} />
        <div className="card space-y-2 p-4 text-xs text-slate-400">
          <div>
            <span className="text-slate-500">Audio:</span>{' '}
            {audio ? `${audio.codec?.toUpperCase()} ${audio.channels}ch ${audio.title ? `· ${audio.title}` : ''}` : '—'}
          </div>
          <div>
            <span className="text-slate-500">Size:</span> {movie.size ? `${(movie.size / 1e9).toFixed(1)} GB` : '—'}
          </div>
          <div>
            <span className="text-slate-500">Free in Censored:</span> {movie.censoredFreeGB != null ? `${movie.censoredFreeGB} GB` : '—'}
          </div>
          <div className="break-all">
            <span className="text-slate-500">File:</span> {movie.file}
          </div>
        </div>
      </aside>

      <section className="min-w-0 space-y-6">
        <div>
          <h1 className="text-2xl font-semibold">
            {movie.title} <span className="font-normal text-slate-500">{movie.year}</span>
          </h1>
          {movie.contentRating && <div className="mt-1 text-sm text-slate-400">{movie.contentRating}</div>}
          {movie.summary && <p className="mt-3 max-w-3xl text-sm text-slate-400">{movie.summary}</p>}
        </div>

        <ErrorBox error={error} />
        {!movie.fileReachable && (
          <ErrorBox error={`The movie file isn't reachable inside the container (${movie.containerPath}). Check PATH_MAPS and the movies volume in docker-compose.`} />
        )}

        {movie.jobs?.length > 0 && (
          <div className="card divide-y divide-slate-800">
            {movie.jobs.map((j) => (
              <Link key={j.id} to={`/jobs/${j.id}`} className="flex items-center gap-3 px-4 py-3 text-sm hover:bg-slate-800/40">
                <StatusBadge status={j.status} />
                <span className="truncate text-slate-300">{j.stage}</span>
                <span className="ml-auto shrink-0 text-xs text-slate-500">
                  {j.username} · {timeAgo(j.created_at)}
                </span>
              </Link>
            ))}
          </div>
        )}

        {movie.fileReachable && (
          <div className="card p-5">
            <div className="flex flex-wrap items-end gap-3">
              <div className="min-w-64 flex-1">
                <label className="mb-1 block text-xs text-slate-400">Subtitle source</label>
                <select className="input" value={option} onChange={(e) => setOption(e.target.value)}>
                  <option value="">Best available{textOptions[0] ? ` (${textOptions[0].label})` : ''}</option>
                  {(movie.subtitleOptions || []).map((o) => (
                    <option key={o.id} value={o.id} disabled={!o.text}>
                      {o.label}
                      {o.sdh ? ' · SDH' : ''}
                      {o.forced ? ' · forced' : ''}
                      {!o.text ? ' · image (not supported yet)' : ''}
                    </option>
                  ))}
                </select>
              </div>
              <button className="btn-primary" onClick={() => startScan(!!scan && scan.status !== 'running')} disabled={scan?.status === 'running'}>
                {scan?.status === 'running' && <Spinner />}
                {scan?.status === 'running' ? 'Scanning…' : scan ? 'Rescan' : 'Scan for language'}
              </button>
            </div>
            {movie.imageSubtitlesOnly && textOptions.length === 0 && (
              <p className="mt-3 text-sm text-amber-300">
                This movie only has image-based subtitles. Let Bazarr download an English .srt next to the file, then scan.
              </p>
            )}
            {scan?.status === 'running' && (
              <p className="mt-3 text-sm text-slate-400">Reading subtitles… embedded tracks in big remuxes can take a minute.</p>
            )}
            {scan?.status === 'failed' && <div className="mt-3"><ErrorBox error={scan.error} /></div>}
          </div>
        )}

        {scan?.status === 'done' && (
          <div className="space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h2 className="text-lg font-semibold">
                  {scan.total} flagged word{scan.total === 1 ? '' : 's'}
                </h2>
                <p className="text-xs text-slate-500">
                  From {scan.subtitleLabel} · {scan.cueCount} lines · the censor step also catches swears spoken near flagged lines that the subtitles left out.
                </p>
              </div>
              <RevealToggle />
            </div>

            {wordgroups && (
              <div className="flex flex-wrap gap-2">
                {wordgroups.presets.map((p) => (
                  <button key={p.id} className="btn-ghost py-1.5 text-xs" onClick={() => setSelected(new Set(p.groups))}>
                    {p.label}
                  </button>
                ))}
                <button className="btn-ghost py-1.5 text-xs" onClick={() => setSelected(new Set())}>
                  Clear
                </button>
              </div>
            )}

            <div className="card divide-y divide-slate-800">
              {groupsWithHits.map((g) => (
                <div key={g.id}>
                  <div className="flex items-center gap-3 px-4 py-3">
                    <input type="checkbox" className="h-4 w-4 accent-amber-500" checked={selected.has(g.id)} onChange={() => toggle(g.id)} />
                    <button className="flex flex-1 items-center gap-3 text-left" onClick={() => setExpanded(expanded === g.id ? null : g.id)}>
                      <span className="font-medium">{g.label}</span>
                      <span className="rounded-full bg-slate-800 px-2 py-0.5 text-xs text-slate-300">{g.count}</span>
                      <span className="hidden truncate text-xs text-slate-500 sm:inline">
                        {Object.entries(g.variants)
                          .sort((a, b) => b[1] - a[1])
                          .slice(0, 6)
                          .map(([w, n]) => (
                            <span key={w} className="mr-3">
                              <Word text={w} /> ×{n}
                            </span>
                          ))}
                      </span>
                      <span className="ml-auto text-xs text-slate-500">{expanded === g.id ? 'Hide' : 'Show lines'}</span>
                    </button>
                  </div>
                  {expanded === g.id && (
                    <ul className="space-y-1 bg-slate-950/40 px-4 py-3 text-sm">
                      {scan.hits
                        .filter((h) => h.group === g.id)
                        .map((h, i) => (
                          <li key={i} className="flex gap-3">
                            <span className="w-16 shrink-0 font-mono text-xs text-slate-500">{fmtTime(h.start)}</span>
                            <span className="text-slate-300">
                              <Line line={h.line} cs={h.cs} ce={h.ce} />
                            </span>
                          </li>
                        ))}
                    </ul>
                  )}
                </div>
              ))}
              {groupsWithHits.length === 0 && <div className="px-4 py-6 text-sm text-slate-400">No flagged words in these subtitles.</div>}
              {groupsWithout.length > 0 && (
                <div className="flex flex-wrap items-center gap-x-4 gap-y-2 px-4 py-3 text-xs text-slate-500">
                  <span>Not in these subtitles (only caught if spoken near other flagged lines):</span>
                  {groupsWithout.map((g) => (
                    <label key={g.id} className="flex items-center gap-1.5">
                      <input type="checkbox" className="accent-amber-500" checked={selected.has(g.id)} onChange={() => toggle(g.id)} />
                      {g.label}
                    </label>
                  ))}
                </div>
              )}
            </div>

            <div className="card grid gap-4 p-5 md:grid-cols-[1fr_auto]">
              <div className="space-y-3">
                <div>
                  <label className="mb-1 block text-xs text-slate-400">Extra words (comma separated, * for wildcard)</label>
                  <input className="input" placeholder="frick*, heck" value={custom} onChange={(e) => setCustom(e.target.value)} />
                </div>
                <div className="flex gap-4 text-sm">
                  {[
                    ['mute', 'Mute (recommended)'],
                    ['bleep', 'Bleep'],
                  ].map(([v, label]) => (
                    <label key={v} className="flex items-center gap-2">
                      <input type="radio" name="style" className="accent-amber-500" checked={style === v} onChange={() => setStyle(v)} />
                      {label}
                    </label>
                  ))}
                </div>
                <div>
                  <div className="mb-1 text-xs text-slate-400">Echo of the word in the other speakers</div>
                  <div className="flex flex-wrap gap-x-4 gap-y-1 text-sm">
                    {[
                      ['deep', 'Remove the voice (best, a few extra minutes)'],
                      ['duck', 'Turn other speakers down briefly'],
                      ['off', 'Mute the dialogue channel only'],
                    ].map(([v, label]) => (
                      <label key={v} className="flex items-center gap-2">
                        <input type="radio" name="echo" className="accent-amber-500" checked={echo === v} onChange={() => setEcho(v)} />
                        {label}
                      </label>
                    ))}
                  </div>
                </div>
              </div>
              <div className="flex flex-col items-end justify-end gap-2">
                <div className="text-sm text-slate-400">
                  {selectedHits.length} subtitle hit{selectedHits.length === 1 ? '' : 's'} selected
                </div>
                <button
                  className="btn-primary"
                  disabled={creating || !!activeJob || (selected.size === 0 && customWords.length === 0)}
                  onClick={createJob}
                >
                  {creating && <Spinner />} Create censored copy
                </button>
                {activeJob && <span className="text-xs text-amber-300">A job for this movie is already in progress.</span>}
              </div>
            </div>
          </div>
        )}
      </section>
    </div>
  )
}
