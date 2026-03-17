import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation } from '@tanstack/react-query'
import toast from 'react-hot-toast'
import UploadForm from '../components/forms/UploadForm'
import { validateUpload, createExperiment } from '../services/api'
import type { UploadResponse, ValidationResponse } from '../types'

export default function UploadPage() {
  const navigate = useNavigate()
  const [upload, setUpload] = useState<UploadResponse | null>(null)
  const [sequenceColumn, setSequenceColumn] = useState('')
  const [nValue, setNValue] = useState(60)
  const [kValue, setKValue] = useState(6)
  const [expName, setExpName] = useState('')
  const [validation, setValidation] = useState<ValidationResponse | null>(null)

  const validateMutation = useMutation({
    mutationFn: () => validateUpload(upload!.upload_id, sequenceColumn, nValue, kValue),
    onSuccess: ({ data }) => setValidation(data),
    onError: () => toast.error('Validation failed'),
  })

  const createMutation = useMutation({
    mutationFn: () =>
      createExperiment({
        name: expName,
        upload_id: upload!.upload_id,
        sequence_column: sequenceColumn,
        n_value: nValue,
        k_value: kValue,
      }),
    onSuccess: ({ data }) => {
      toast.success('Experiment created')
      navigate(`/experiments/${data.id}`)
    },
    onError: () => toast.error('Could not create experiment'),
  })

  if (!upload) {
    return (
      <div className="max-w-2xl mx-auto">
        <h1 className="text-2xl font-bold mb-6">Upload Dataset</h1>
        <UploadForm onSuccess={setUpload} />
      </div>
    )
  }

  const numericCols = upload.columns.filter((c) => c.is_numeric)

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <h1 className="text-2xl font-bold">Configure Experiment</h1>

      <div className="bg-white rounded-lg border border-gray-200 p-5 space-y-4">
        <p className="text-sm text-gray-600">
          <span className="font-medium">{upload.filename}</span> — {upload.row_count} rows,{' '}
          {upload.column_count} columns
        </p>

        <div>
          <label className="block text-sm font-medium mb-1">Experiment name</label>
          <input
            value={expName}
            onChange={(e) => setExpName(e.target.value)}
            placeholder="My experiment"
            className="w-full border border-gray-300 rounded px-3 py-2 text-sm"
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Sequence column</label>
          <select
            value={sequenceColumn}
            onChange={(e) => setSequenceColumn(e.target.value)}
            className="w-full border border-gray-300 rounded px-3 py-2 text-sm"
          >
            <option value="">Select a column…</option>
            {numericCols.map((c) => (
              <option key={c.name} value={c.name}>
                {c.name} ({c.dtype})
              </option>
            ))}
          </select>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium mb-1">N (domain max)</label>
            <input
              type="number"
              min={1}
              value={nValue}
              onChange={(e) => setNValue(+e.target.value)}
              className="w-full border border-gray-300 rounded px-3 py-2 text-sm"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">k (draw size)</label>
            <input
              type="number"
              min={1}
              value={kValue}
              onChange={(e) => setKValue(+e.target.value)}
              className="w-full border border-gray-300 rounded px-3 py-2 text-sm"
            />
          </div>
        </div>

        <button
          onClick={() => validateMutation.mutate()}
          disabled={!sequenceColumn || validateMutation.isPending}
          className="px-4 py-2 bg-gray-100 border border-gray-300 rounded text-sm hover:bg-gray-200 disabled:opacity-50"
        >
          Validate
        </button>

        {validation && (
          <div className={`rounded p-3 text-sm ${validation.valid ? 'bg-green-50' : 'bg-red-50'}`}>
            <p className="font-medium">{validation.valid ? 'Valid' : 'Issues found'}</p>
            <p className="text-xs text-gray-600">Usable rows: {validation.usable_rows}</p>
            {validation.issues.map((issue, i) => (
              <p key={i} className={`text-xs mt-1 ${issue.level === 'error' ? 'text-red-600' : 'text-yellow-700'}`}>
                [{issue.level}] {issue.message}
              </p>
            ))}
          </div>
        )}
      </div>

      <div className="flex gap-3">
        <button
          onClick={() => setUpload(null)}
          className="px-4 py-2 border border-gray-300 rounded text-sm hover:bg-gray-50"
        >
          Back
        </button>
        <button
          onClick={() => createMutation.mutate()}
          disabled={!expName || !sequenceColumn || createMutation.isPending}
          className="px-4 py-2 bg-primary-600 text-white rounded text-sm hover:bg-primary-700 disabled:opacity-50"
        >
          Create Experiment
        </button>
      </div>
    </div>
  )
}
