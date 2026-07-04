import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'

createRoot(document.getElementById('root')!).render(
  
  <StrictMode>
    <BrowserRouter>
    <App/>
    </BrowserRouter>
   
  </StrictMode>,
)
