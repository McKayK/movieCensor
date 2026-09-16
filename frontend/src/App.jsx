import { createContext, useCallback, useContext, useEffect, useState } from 'react'
import { Link, NavLink, Navigate, Route, Routes, useLocation } from 'react-router-dom'
import { api } from './api'
import { Spinner } from './components/ui'
import AuthCallback from './pages/AuthCallback'
import Job from './pages/Job'
import Jobs from './pages/Jobs'
import Library from './pages/Library'
import Login from './pages/Login'
import Movie from './pages/Movie'

const UserContext = createContext({ user: null, refresh: () => {} })
export const useUser = () => useContext(UserContext)

export default function App() {
  const [user, setUser] = useState(undefined)
  const location = useLocation()

  const refresh = useCallback(() => {
    return api('/api/auth/me')
      .then(setUser)
      .catch(() => setUser(null))
  }, [])

  useEffect(() => {
    refresh()
  }, [refresh])

  if (location.pathname === '/auth/callback') {
    return (
      <UserContext.Provider value={{ user, refresh }}>
        <AuthCallback />
      </UserContext.Provider>
    )
  }
  if (user === undefined) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <Spinner className="h-6 w-6" />
      </div>
    )
  }
  if (!user) return <Login />

  return (
    <UserContext.Provider value={{ user, refresh }}>
      <Shell>
        <Routes>
          <Route path="/" element={<Library />} />
          <Route path="/movie/:key" element={<Movie />} />
          <Route path="/jobs" element={<Jobs />} />
          <Route path="/jobs/:id" element={<Job />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Shell>
    </UserContext.Provider>
  )
}

function Shell({ children }) {
  const { user } = useUser()
  const logout = async () => {
    await api('/api/auth/logout', { method: 'POST' }).catch(() => {})
    window.location.href = '/'
  }
  const tab = ({ isActive }) =>
    `rounded-md px-3 py-1.5 text-sm font-medium ${isActive ? 'bg-slate-800 text-white' : 'text-slate-400 hover:text-white'}`
  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-20 border-b border-slate-800 bg-slate-950/90 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center gap-4 px-4 py-3">
          <Link to="/" className="flex items-center gap-2 font-semibold tracking-tight">
            <img src="/favicon.svg" alt="" className="h-7 w-7" />
            <span>Movie Censor</span>
          </Link>
          <nav className="flex gap-1">
            <NavLink to="/" end className={tab}>
              Library
            </NavLink>
            <NavLink to="/jobs" className={tab}>
              Jobs
            </NavLink>
          </nav>
          <div className="ml-auto flex items-center gap-3 text-sm text-slate-400">
            {user.thumb && <img src={user.thumb} alt="" className="h-7 w-7 rounded-full" />}
            <span className="hidden sm:inline">{user.username}</span>
            <button onClick={logout} className="text-slate-500 hover:text-white">
              Sign out
            </button>
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-4 py-6">{children}</main>
    </div>
  )
}
