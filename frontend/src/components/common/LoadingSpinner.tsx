interface Props {
  size?: 'sm' | 'md' | 'lg'
  label?: string
}

const SIZE = { sm: 'h-4 w-4', md: 'h-8 w-8', lg: 'h-12 w-12' }

export default function LoadingSpinner({ size = 'md', label }: Props) {
  return (
    <div className="flex flex-col items-center gap-2">
      <div
        className={`${SIZE[size]} animate-spin rounded-full border-4 border-primary-200 border-t-primary-600`}
        role="status"
        aria-label={label ?? 'Loading'}
      />
      {label && <p className="text-sm text-gray-500">{label}</p>}
    </div>
  )
}
