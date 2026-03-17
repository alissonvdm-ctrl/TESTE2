import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { listExperiments } from '../services/api'
import LoadingSpinner from '../components/common/LoadingSpinner'
import ErrorMessage from '../components/common/ErrorMessage'
import StatusBadge from '../components/common/StatusBadge'
import { formatDistanceToNow } from 'date-fns'

export default function ExperimentsPage() {
  const { data, isLoading, isError, refetch } = useQuery({
    queryKey: ['experiments'],
    queryFn: () => listExperiments(),
    select: (r) => r.data,
  })

  if (isLoading) return <div className="flex justify-center mt-20"><LoadingSpinner label="Loading experiments…" /></div>
  if (isError) return <ErrorMessage message="Failed to load experiments." onRetry={refetch} />

  const experiments = data?.items ?? []

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Experiments</h1>
        <Link
          to="/upload"
          className="px-4 py-2 bg-primary-600 text-white rounded text-sm hover:bg-primary-700"
        >
          New experiment
        </Link>
      </div>

      {experiments.length === 0 ? (
        <div className="text-center py-20 text-gray-500">
          <p className="text-lg">No experiments yet.</p>
          <Link to="/upload" className="text-primary-600 underline mt-2 inline-block">
            Upload a dataset to get started
          </Link>
        </div>
      ) : (
        <div className="space-y-3">
          {experiments.map((exp) => (
            <Link
              key={exp.id}
              to={`/experiments/${exp.id}`}
              className="block bg-white border border-gray-200 rounded-lg p-4 hover:shadow-sm transition-shadow"
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium text-gray-900">{exp.name}</p>
                  {exp.description && (
                    <p className="text-sm text-gray-500 mt-0.5">{exp.description}</p>
                  )}
                  <p className="text-xs text-gray-400 mt-1">
                    Created{' '}
                    {formatDistanceToNow(new Date(exp.created_at), { addSuffix: true })}
                  </p>
                </div>
                <StatusBadge status={exp.status} />
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  )
}
