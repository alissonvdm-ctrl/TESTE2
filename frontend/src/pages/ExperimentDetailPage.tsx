import { useParams, useNavigate } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import toast from 'react-hot-toast'
import {
  getExperiment,
  getExperimentStatus,
  startExperiment,
  getChangepoints,
  getEnsemble,
  deleteExperiment,
} from '../services/api'
import { useWebSocket } from '../hooks/useWebSocket'
import { useExperimentStore } from '../store/useExperimentStore'
import LoadingSpinner from '../components/common/LoadingSpinner'
import ErrorMessage from '../components/common/ErrorMessage'
import StatusBadge from '../components/common/StatusBadge'
import ChangepointChart from '../components/charts/ChangepointChart'
import type { ProgressMessage } from '../types'

export default function ExperimentDetailPage() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const qc = useQueryClient()
  const { updateProgress, progress } = useExperimentStore()

  const expQuery = useQuery({
    queryKey: ['experiment', id],
    queryFn: () => getExperiment(id!),
    select: (r) => r.data,
  })

  const statusQuery = useQuery({
    queryKey: ['experiment-status', id],
    queryFn: () => getExperimentStatus(id!),
    select: (r) => r.data,
    refetchInterval: expQuery.data?.status === 'running' ? 3000 : false,
  })

  const changepointsQuery = useQuery({
    queryKey: ['changepoints', id],
    queryFn: () => getChangepoints(id!),
    select: (r) => r.data,
    enabled: expQuery.data?.status === 'completed',
  })

  const ensembleQuery = useQuery({
    queryKey: ['ensemble', id],
    queryFn: () => getEnsemble(id!),
    select: (r) => r.data,
    enabled: expQuery.data?.status === 'completed',
  })

  const startMutation = useMutation({
    mutationFn: () => startExperiment(id!),
    onSuccess: () => {
      toast.success('Analysis started')
      qc.invalidateQueries({ queryKey: ['experiment', id] })
    },
  })

  const deleteMutation = useMutation({
    mutationFn: () => deleteExperiment(id!),
    onSuccess: () => {
      toast.success('Experiment deleted')
      navigate('/experiments')
    },
  })

  // WebSocket for live progress
  useWebSocket(id ?? null, {
    onMessage: (msg: ProgressMessage) => {
      updateProgress(msg)
      if (msg.status === 'completed' || msg.status === 'failed') {
        qc.invalidateQueries({ queryKey: ['experiment', id] })
      }
    },
  })

  const liveProgress = id ? progress[id] : undefined

  if (expQuery.isLoading) return <div className="flex justify-center mt-20"><LoadingSpinner /></div>
  if (expQuery.isError) return <ErrorMessage message="Experiment not found." />

  const exp = expQuery.data!

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold">{exp.name}</h1>
          {exp.description && <p className="text-gray-600 mt-1">{exp.description}</p>}
          <div className="flex items-center gap-3 mt-2">
            <StatusBadge status={exp.status} />
            <span className="text-xs text-gray-400">ID: {exp.id}</span>
          </div>
        </div>
        <div className="flex gap-2">
          {exp.status === 'pending' && (
            <button
              onClick={() => startMutation.mutate()}
              disabled={startMutation.isPending}
              className="px-4 py-2 bg-primary-600 text-white rounded text-sm hover:bg-primary-700 disabled:opacity-50"
            >
              Start analysis
            </button>
          )}
          <button
            onClick={() => {
              if (confirm('Delete this experiment?')) deleteMutation.mutate()
            }}
            className="px-3 py-2 border border-red-300 text-red-600 rounded text-sm hover:bg-red-50"
          >
            Delete
          </button>
        </div>
      </div>

      {/* Config */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <h2 className="font-semibold mb-3">Configuration</h2>
        <dl className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
          {[
            ['Column', exp.sequence_column],
            ['N (domain max)', exp.n_value ?? '—'],
            ['k (draw size)', exp.k_value ?? '—'],
            ['Upload ID', exp.upload_id.slice(0, 8) + '…'],
          ].map(([label, value]) => (
            <div key={String(label)}>
              <dt className="text-gray-500">{label}</dt>
              <dd className="font-medium">{value}</dd>
            </div>
          ))}
        </dl>
      </div>

      {/* Live progress */}
      {liveProgress && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <p className="text-sm font-medium text-blue-800">
            {liveProgress.stage} — {liveProgress.message}
          </p>
          <div className="mt-2 h-2 bg-blue-200 rounded-full overflow-hidden">
            <div
              className="h-full bg-blue-600 transition-all"
              style={{ width: `${liveProgress.progress_pct}%` }}
            />
          </div>
          <p className="text-xs text-blue-700 mt-1">{liveProgress.progress_pct.toFixed(0)}%</p>
        </div>
      )}

      {/* Status */}
      {statusQuery.data && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <h2 className="font-semibold mb-2">Pipeline Status</h2>
          <p className="text-sm text-gray-600">{statusQuery.data.message}</p>
          {statusQuery.data.error && (
            <p className="text-sm text-red-600 mt-1">Error: {statusQuery.data.error}</p>
          )}
        </div>
      )}

      {/* Ensemble result */}
      {ensembleQuery.data && ensembleQuery.data.weights.length > 0 && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <h2 className="font-semibold mb-3">Ensemble Result</h2>
          <p className="text-sm text-gray-600 mb-2">
            Strategy: <span className="font-medium">{ensembleQuery.data.strategy}</span>
          </p>
          <div className="flex flex-wrap gap-2">
            {ensembleQuery.data.weights.map((w) => (
              <span
                key={w.model_name}
                className="px-2 py-1 bg-green-50 border border-green-200 rounded text-xs"
              >
                {w.model_name}: {(w.weight * 100).toFixed(1)}%
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Change-points */}
      {changepointsQuery.data && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <h2 className="font-semibold mb-3">
            Change-Point Analysis — {changepointsQuery.data.n_changepoints} breakpoints detected
          </h2>
          <ChangepointChart regimes={changepointsQuery.data.regimes} />
        </div>
      )}
    </div>
  )
}
