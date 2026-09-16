import { createContext, useContext, useState } from 'react'

const RevealContext = createContext({ reveal: false, setReveal: () => {} })

function readReveal() {
  try {
    return localStorage.getItem('reveal-words') === '1'
  } catch {
    return false
  }
}

export function RevealProvider({ children }) {
  const [reveal, setRevealState] = useState(readReveal)
  const setReveal = (v) => {
    setRevealState(v)
    try {
      localStorage.setItem('reveal-words', v ? '1' : '0')
    } catch {
      /* private mode */
    }
  }
  return <RevealContext.Provider value={{ reveal, setReveal }}>{children}</RevealContext.Provider>
}

export const useReveal = () => useContext(RevealContext)

export function maskWord(word) {
  if (!word) return ''
  return word.replace(/[A-Za-z]/g, (c, i) => (i === 0 || !/[A-Za-z]/.test(word[i - 1] ?? ' ') ? c : '*'))
}

/** Mask the given char span (or every case-insensitive occurrence of `word`) inside a subtitle line. */
export function maskLine(line, { cs, ce, word } = {}) {
  if (!line) return ''
  if (cs != null && ce != null) {
    return line.slice(0, cs) + maskWord(line.slice(cs, ce)) + line.slice(ce)
  }
  if (word) {
    const esc = word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
    return line.replace(new RegExp(esc, 'ig'), (m) => maskWord(m))
  }
  return line
}

export function Word({ text }) {
  const { reveal } = useReveal()
  return <span className="font-semibold text-amber-300">{reveal ? text : maskWord(text)}</span>
}

export function Line({ line, cs, ce, word }) {
  const { reveal } = useReveal()
  return <span className="whitespace-pre-line">{reveal ? line : maskLine(line, { cs, ce, word })}</span>
}

export function RevealToggle() {
  const { reveal, setReveal } = useReveal()
  return (
    <label className="flex cursor-pointer items-center gap-2 text-xs text-slate-400 select-none">
      <input type="checkbox" className="accent-amber-500" checked={reveal} onChange={(e) => setReveal(e.target.checked)} />
      Show words uncensored
    </label>
  )
}
