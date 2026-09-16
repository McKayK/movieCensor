import { useState } from 'react'
import { api } from '../api'
import { ErrorBox, Spinner } from '../components/ui'

export default function Login() {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  const signIn = async () => {
    setBusy(true)
    setError(null)
    try {
      const { url } = await api('/api/auth/login')
      window.location.href = url
    } catch (e) {
      setError(e)
      setBusy(false)
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center px-4">
      <div className="card w-full max-w-sm p-8 text-center">
        <img src="/favicon.svg" alt="" className="mx-auto mb-4 h-12 w-12" />
        <h1 className="text-xl font-semibold">Movie Censor</h1>
        <p className="mt-2 text-sm text-slate-400">Make family-friendly copies of movies in the Plex library.</p>
        <button onClick={signIn} disabled={busy} className="btn-primary mt-6 w-full">
          {busy && <Spinner />} Sign in with Plex
        </button>
        <div className="mt-4">
          <ErrorBox error={error} />
        </div>
      </div>
    </div>
  )
}
