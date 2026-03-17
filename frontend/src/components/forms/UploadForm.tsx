import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import toast from 'react-hot-toast'
import { uploadExcel } from '../../services/api'
import type { UploadResponse } from '../../types'
import LoadingSpinner from '../common/LoadingSpinner'

interface Props {
  onSuccess: (result: UploadResponse) => void
}

export default function UploadForm({ onSuccess }: Props) {
  const [uploading, setUploading] = useState(false)

  const onDrop = useCallback(
    async (files: File[]) => {
      const file = files[0]
      if (!file) return
      setUploading(true)
      try {
        const { data } = await uploadExcel(file)
        toast.success(`Uploaded ${data.row_count} rows`)
        onSuccess(data)
      } catch (err: unknown) {
        const msg = (err as { response?: { data?: { detail?: string } } })
          ?.response?.data?.detail ?? 'Upload failed'
        toast.error(msg)
      } finally {
        setUploading(false)
      }
    },
    [onSuccess],
  )

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls'],
    },
    maxFiles: 1,
    disabled: uploading,
  })

  return (
    <div
      {...getRootProps()}
      className={`border-2 border-dashed rounded-lg p-10 text-center cursor-pointer transition-colors ${
        isDragActive ? 'border-primary-500 bg-primary-50' : 'border-gray-300 hover:border-primary-400'
      }`}
    >
      <input {...getInputProps()} />
      {uploading ? (
        <LoadingSpinner label="Uploading…" />
      ) : (
        <>
          <p className="text-gray-600">
            {isDragActive ? 'Drop the file here…' : 'Drag & drop an Excel file, or click to select'}
          </p>
          <p className="text-xs text-gray-400 mt-1">.xlsx / .xls — max 100 MB</p>
        </>
      )}
    </div>
  )
}
