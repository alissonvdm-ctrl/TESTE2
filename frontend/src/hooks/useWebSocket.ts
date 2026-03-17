import { useEffect, useRef, useCallback } from 'react'
import type { ProgressMessage } from '../types'

const WS_BASE = (import.meta.env.VITE_WS_URL ?? 'ws://localhost:8000').replace(/\/$/, '')

interface UseWebSocketOptions {
  onMessage?: (msg: ProgressMessage) => void
  onOpen?: () => void
  onClose?: () => void
  onError?: (e: Event) => void
}

export function useWebSocket(experimentId: string | null, options: UseWebSocketOptions = {}) {
  const wsRef = useRef<WebSocket | null>(null)
  const optsRef = useRef(options)
  optsRef.current = options

  const connect = useCallback(() => {
    if (!experimentId) return
    const url = `${WS_BASE}/analysis/ws/${experimentId}`
    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = () => optsRef.current.onOpen?.()
    ws.onclose = () => optsRef.current.onClose?.()
    ws.onerror = (e) => optsRef.current.onError?.(e)
    ws.onmessage = (e) => {
      try {
        const msg: ProgressMessage = JSON.parse(e.data)
        optsRef.current.onMessage?.(msg)
      } catch {
        // ignore parse errors
      }
    }
  }, [experimentId])

  const disconnect = useCallback(() => {
    wsRef.current?.close()
    wsRef.current = null
  }, [])

  useEffect(() => {
    connect()
    return disconnect
  }, [connect, disconnect])

  return { disconnect }
}
