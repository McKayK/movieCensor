import { useEffect, useState } from 'react'
import { api } from '../api'
import { Spinner } from '../components/ui'

export default function AuthCallback() {
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false
    const attempt = async (tries) => {
      try {
        await api('/api/auth/callback')
        if (!cancelled) window.location.replace('/')
      } catch (e) {
        // Plex sometimes redirects a moment before the PIN is marked approved.
        if (e.status === 401 && tries < 30 && !cancelled) {
          setTimeout(() => attempt(tries + 1), 1000)
        } else if (!cancelled) {
          setError(e.message)
        }
      }
    }
    attempt(0)
    return () => {
      cancelled = true
    }
  }, [])

  return (
    <div className="flex min-h-screen items-center justify-center px-4">
      <div className="card w-full max-w-sm p-8 text-center">
        {error ? (
          <>
            <p className="text-rose-300">{error}</p>
            <a href="/" className="btn-ghost mt-6">
              Back to sign in
            </a>
          </>
        ) : (
          <div className="flex items-center justify-center gap-3 text-slate-300">
            <Spinner /> Finishing Plex sign-in…
          </div>
        )}
      </div>
    </div>
  )
}
