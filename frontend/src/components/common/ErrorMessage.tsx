interface Props {
  title?: string
  message: string
  onRetry?: () => void
}

export default function ErrorMessage({ title = 'Error', message, onRetry }: Props) {
  return (
    <div className="rounded-lg bg-red-50 border border-red-200 p-4">
      <p className="font-semibold text-red-800">{title}</p>
      <p className="text-sm text-red-700 mt-1">{message}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="mt-2 text-sm text-red-600 underline hover:text-red-800"
        >
          Try again
        </button>
      )}
    </div>
  )
}
