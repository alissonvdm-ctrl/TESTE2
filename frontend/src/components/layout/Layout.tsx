import { Outlet } from 'react-router-dom'
import Navbar from './Navbar'

export default function Layout() {
  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <main className="flex-1 container mx-auto px-4 py-6 max-w-7xl">
        <Outlet />
      </main>
      <footer className="border-t border-gray-200 py-4 text-center text-sm text-gray-500">
        Numeric Sequence Analyzer
      </footer>
    </div>
  )
}
