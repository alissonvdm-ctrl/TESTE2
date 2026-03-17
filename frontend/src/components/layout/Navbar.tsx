import { Link, useLocation } from 'react-router-dom'
import clsx from 'clsx'

const NAV_LINKS = [
  { to: '/', label: 'Home' },
  { to: '/upload', label: 'Upload' },
  { to: '/experiments', label: 'Experiments' },
]

export default function Navbar() {
  const { pathname } = useLocation()

  return (
    <nav className="bg-white border-b border-gray-200 shadow-sm">
      <div className="container mx-auto px-4 max-w-7xl flex items-center h-14">
        <Link to="/" className="font-bold text-primary-700 text-lg mr-8">
          NSA
        </Link>
        <div className="flex gap-1">
          {NAV_LINKS.map(({ to, label }) => (
            <Link
              key={to}
              to={to}
              className={clsx(
                'px-3 py-2 rounded text-sm font-medium transition-colors',
                pathname === to
                  ? 'bg-primary-50 text-primary-700'
                  : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100',
              )}
            >
              {label}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  )
}
