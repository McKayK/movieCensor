import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App.jsx'
import { RevealProvider } from './mask.jsx'
import './index.css'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <BrowserRouter>
      <RevealProvider>
        <App />
      </RevealProvider>
    </BrowserRouter>
  </StrictMode>,
)
