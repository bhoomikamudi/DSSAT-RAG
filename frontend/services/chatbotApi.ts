/**
 * API service for the DSSAT RAG backend.
 */

import axios, { AxiosError, AxiosInstance } from 'axios'
import {
  ChatRequest,
  ChatResponse,
  ApiError,
  ManagementVariables,
  MapPoint,
  SpatialConfig,
  SpatialCoverage,
} from '@/types/chat'

class ChatbotApiService {
  private client: AxiosInstance
  private baseURL: string

  constructor(baseURL: string = 'http://localhost:8005') {
    this.baseURL = baseURL

    this.client = axios.create({
      baseURL,
      timeout: 60000,
      headers: {
        'Content-Type': 'application/json',
      },
    })
  }

  /**
   * Send a user question to the backend.
   */
  async sendMessage(
    userQuery: string,
    sessionId?: string,
    location?: { point: MapPoint | null; radiusKm: number | null }
  ): Promise<ChatResponse> {
    if (!userQuery.trim()) {
      throw {
        message: 'Query cannot be empty',
        code: 'ERR_EMPTY_QUERY',
      } as ApiError
    }

    try {
      const request: ChatRequest = {
        message: userQuery.trim(),
        session_id: sessionId,
      }

      // The radius is sent even without a map point so that a place named
      // in the question ("near Kitale") uses the user's chosen radius.
      if (location?.radiusKm) {
        request.radius_km = location.radiusKm
      }
      if (location?.point) {
        request.latitude = location.point.latitude
        request.longitude = location.point.longitude
      }

      const response = await this.client.post<ChatResponse>(
        '/api/v1/chat/',
        request
      )

      return response.data
    } catch (error) {
      throw this.handleError(error)
    }
  }

  /**
   * Default/maximum radius and whether place lookup is enabled.
   */
  async getSpatialConfig(): Promise<SpatialConfig> {
    const response = await this.client.get<SpatialConfig>('/api/v1/spatial/config')
    return response.data
  }

  /**
   * Management fields (cultivar, planting date, ...) and the values in the data.
   */
  async getManagementVariables(): Promise<ManagementVariables> {
    const response = await this.client.get<ManagementVariables>('/api/v1/dataset/management-variables')
    return response.data
  }

  /**
   * Where ingested simulation data exists, for drawing on the map.
   */
  async getSpatialCoverage(): Promise<SpatialCoverage> {
    const response = await this.client.get<SpatialCoverage>('/api/v1/spatial/coverage')
    return response.data
  }

  /**
   * Check whether the backend is available.
   */
  async healthCheck(): Promise<boolean> {
    try {
      await this.client.get('/', {
        timeout: 5000,
      })

      return true
    } catch {
      return false
    }
  }

  /**
   * Change the backend URL.
   */
  setBaseURL(newBaseURL: string): void {
    this.baseURL = newBaseURL
    this.client.defaults.baseURL = newBaseURL
  }

  /**
   * Get the current backend URL.
   */
  getBaseURL(): string {
    return this.baseURL
  }

  /**
   * Convert request errors into a consistent format.
   */
  private handleError(error: unknown): ApiError {
    if (axios.isAxiosError(error)) {
      const axiosError = error as AxiosError<{
        message?: string
        detail?: string
      }>

      if (axiosError.response) {
        return {
          message:
            axiosError.response.data?.message ||
            axiosError.response.data?.detail ||
            `Server error: ${axiosError.response.status}`,
          code: `ERR_${axiosError.response.status}`,
          details: axiosError.response.data,
        }
      }

      if (axiosError.request) {
        return {
          message:
            'No response from the DSSAT backend. Please ensure it is running on http://localhost:8005.',
          code: 'ERR_NO_RESPONSE',
        }
      }
    }

    return {
      message:
        error instanceof Error
          ? error.message
          : 'An unexpected error occurred.',
      code: 'ERR_UNKNOWN',
    }
  }
}

export const chatbotApiService = new ChatbotApiService(
  process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8005'
)

export default chatbotApiService