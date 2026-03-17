import { Link } from 'react-router-dom'

const FEATURES = [
  {
    title: 'Upload & Validate',
    desc: 'Import Excel files with numeric sequences and validate data quality.',
    to: '/upload',
    color: 'bg-blue-50 border-blue-200',
  },
  {
    title: 'Run Analysis',
    desc: 'Statistical diagnostics, change-point detection, and ML forecasting.',
    to: '/experiments',
    color: 'bg-purple-50 border-purple-200',
  },
  {
    title: 'Ensemble Predictions',
    desc: 'Combine multiple models into a weighted ensemble for best results.',
    to: '/experiments',
    color: 'bg-green-50 border-green-200',
  },
]

export default function HomePage() {
  return (
    <div className="max-w-3xl mx-auto mt-12 text-center">
      <h1 className="text-4xl font-bold text-gray-900 mb-4">Numeric Sequence Analyzer</h1>
      <p className="text-lg text-gray-600 mb-10">
        Upload numeric sequence data, run statistical and machine-learning analyses,
        and generate ensemble predictions with confidence scores.
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-left mb-10">
        {FEATURES.map(({ title, desc, to, color }) => (
          <Link
            key={title}
            to={to}
            className={`rounded-lg border p-5 ${color} hover:shadow-md transition-shadow`}
          >
            <h2 className="font-semibold text-gray-800 mb-1">{title}</h2>
            <p className="text-sm text-gray-600">{desc}</p>
          </Link>
        ))}
      </div>

      <Link
        to="/upload"
        className="inline-block px-6 py-3 bg-primary-600 text-white rounded-lg font-medium hover:bg-primary-700 transition-colors"
      >
        Get started
      </Link>
    </div>
  )
}
