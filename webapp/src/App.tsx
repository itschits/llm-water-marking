import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Detector from './pages/Detector'
import Charts from './pages/Charts'
import RunExperiment from './pages/RunExperiment'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="run" element={<RunExperiment />} />
          <Route path="detect" element={<Detector />} />
          <Route path="charts" element={<Charts />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
