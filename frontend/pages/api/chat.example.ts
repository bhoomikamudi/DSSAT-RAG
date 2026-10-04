/**
 * Optional Next.js API route proxy for the DSSAT backend.
 *
 * This file is not required because the frontend currently calls
 * the backend directly. It is kept here as an optional proxy.
 */

import type { NextApiRequest, NextApiResponse } from 'next'
import axios from 'axios'
import { ChatRequest, ChatResponse } from '@/types/chat'

const BACKEND_CHAT_URL =
  process.env.DSSAT_BACKEND_CHAT_URL ||
  'http://localhost:8005/api/v1/chat/'

const REQUEST_TIMEOUT = 60000

interface ApiErrorResponse {
  error: string
  message: string
  status: number
}

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<ChatResponse | ApiErrorResponse>
): Promise<void> {
  if (req.method !== 'POST') {
    res.status(405).json({
      error: 'Method Not Allowed',
      message: `HTTP ${req.method} is not supported`,
      status: 405,
    })
    return
  }

  try {
    const { message } = req.body as ChatRequest

    if (typeof message !== 'string' || message.trim().length === 0) {
      res.status(400).json({
        error: 'Bad Request',
        message: 'message is required and must be a non-empty string',
        status: 400,
      })
      return
    }

    console.log(`[API] Chat request: ${message.substring(0, 50)}...`)

    const response = await axios.post<ChatResponse>(
      BACKEND_CHAT_URL,
      { message: message.trim() },
      {
        timeout: REQUEST_TIMEOUT,
        headers: {
          'Content-Type': 'application/json',
        },
      }
    )

    console.log(
      `[API] Chat response received: ${response.data.answer.substring(0, 50)}...`
    )

    res.status(200).json(response.data)
  } catch (error) {
    console.error('[API] Error:', error)

    if (axios.isAxiosError(error)) {
      const status = error.response?.status || 500
      const responseData = error.response?.data as
        | { detail?: string; message?: string }
        | undefined

      res.status(status).json({
        error: 'Backend Service Error',
        message:
          responseData?.message ||
          responseData?.detail ||
          error.message ||
          'Backend request failed',
        status,
      })
      return
    }

    res.status(500).json({
      error: 'Internal Server Error',
      message:
        error instanceof Error ? error.message : 'An unexpected error occurred',
      status: 500,
    })
  }
}