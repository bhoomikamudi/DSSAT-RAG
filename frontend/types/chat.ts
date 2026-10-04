/**
 * Chat message types for the application
 */

export type ChatRole = 'user' | 'assistant' | 'system'

export interface Statistics {
  aggregation_type?: string
  metric?: string
  value?: number | null
  count?: number
  stddev?: number | null
  min?: number | null
  max?: number | null
  breakdown?: Record<string, unknown> | null
  unit?: string | null
}

export interface AnalysisPoint {
  x: number | null
  y: number | null
}

export interface AnalysisAxis {
  code: string
  label: string
  unit?: string | null
}

export interface AnalysisSeries {
  name: string
  n: number
  points: AnalysisPoint[]
  fit?: AnalysisPoint[] | null
  correlation?: number | null
  r_squared?: number | null
}

export interface AnalysisBar {
  group: string
  mean: number | null
  std?: number | null
  count: number
}

export interface AnalysisChartData {
  x?: AnalysisAxis | null
  y?: AnalysisAxis | null
  group_by?: string | null
  series?: AnalysisSeries[]
  bars?: AnalysisBar[]
  total_points?: number
  plotted_points?: number
  sampled?: boolean
}

export interface AnalysisGroupResult {
  group?: string | null
  status: 'ok' | 'insufficient_data'
  sample_size: number
  correlation?: number | null
  p_value?: number | null
  coefficients?: Record<string, number | null> | null
  r_squared?: number | null
  message?: string | null
}

export interface AnalysisResult {
  status: 'ok' | 'insufficient_data' | 'unavailable' | 'invalid'
  analysis_type: string
  x_variable?: string | null
  y_variable?: string | null
  group_by?: string | null
  sample_size: number
  correlation?: number | null
  p_value?: number | null
  coefficients?: Record<string, number | null> | null
  r_squared?: number | null
  groups: AnalysisGroupResult[]
  chart_type?: 'scatter' | 'scatter_with_line' | 'scatter_with_curve' | 'bar' | null
  chart_data?: AnalysisChartData | null
  explanation: string
  warnings: string[]
}

/** Radius filter applied to a response (from the backend "spatial" field). */
export interface SpatialScope {
  filter?: {
    latitude: number
    longitude: number
    radius_km: number
    source: 'map' | 'place' | 'coordinates'
    label?: string | null
  }
  description?: string
  simulations?: number
  locations?: number
  note?: string | null
  notice?: string | null
  /** True when the question asked to drop the location filter. */
  cleared?: boolean
}

/** A point chosen on the map, sent with each question. */
export interface MapPoint {
  latitude: number
  longitude: number
}

export interface SpatialConfig {
  default_radius_km: number
  max_radius_km: number
  geocoder_enabled: boolean
}

export interface DataLocation {
  latitude: number
  longitude: number
  simulations: number
}

export interface SpatialCoverage {
  simulations: number
  location_count: number
  locations_truncated: boolean
  bounds: { min_lat: number; max_lat: number; min_lon: number; max_lon: number } | null
  locations: DataLocation[]
}

export interface ChatMessage {
  id: string
  role: ChatRole
  content: string
  timestamp: Date
  isLoading?: boolean
  error?: string
  statistics?: Statistics | null
  analysis?: AnalysisResult | null
  spatial?: SpatialScope | null
  /** Location state sent with this question (user messages only). */
  sentLocation?: SentLocation
  management?: ManagementScope | null
}

/** One management field present in the data, with its stored values. */
export interface ManagementField {
  field: string
  label: string
  values: Array<{ value: string; label: string; simulations: number }>
}

/** GET /api/v1/dataset/management-variables */
export interface ManagementVariables {
  fields: ManagementField[]
  simulations: number
  years: { min: number | null; max: number | null }
  crops: string[]
}

/** Management conditions of the records behind one answer. */
export interface ManagementScope {
  simulations: number
  sentence: string
  requested: Array<{ field: string; label: string; values: string[]; management: boolean }>
  included: Array<{
    field: string
    label: string
    values: string[]
    value_labels: string[]
    available_count: number
    all_available: boolean
    filtered: boolean
    grouped: boolean
  }>
  /** Requested conditions the pipeline could not apply as filters, with the reason. */
  unapplied?: Array<{ key: string; condition: string; phrase: string; reason: string }>
}

/** The map point and radius included in a chat request. */
export interface SentLocation {
  point: MapPoint | null
  radiusKm: number | null
}

export interface ChatRequest {
  message: string
  session_id?: string
  latitude?: number
  longitude?: number
  radius_km?: number
}

export interface ChatResponse {
  answer: string
  statistics?: Statistics | null
  analysis?: AnalysisResult | null
  spatial?: SpatialScope | null
  management?: ManagementScope | null
  semantic_plan?: unknown
  sources?: Array<{
    type: string
    description: string
  }>
  confidence?: string
}

export interface ApiError {
  message: string
  code?: string
  details?: unknown
}

export interface ChatContextState {
  messages: ChatMessage[]
  isLoading: boolean
  error: string | null
}