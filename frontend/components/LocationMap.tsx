/**
 * LocationMap Component
 * Leaflet map showing where simulation data exists; clicking picks the
 * point used for radius filtering. Browser-only: load with next/dynamic
 * and ssr: false.
 */

import React, { useEffect } from 'react'
import {
  MapContainer,
  TileLayer,
  CircleMarker,
  Circle,
  Tooltip,
  useMap,
  useMapEvents,
} from 'react-leaflet'
import type { LatLngBoundsExpression } from 'leaflet'
import { MapPoint, SpatialCoverage } from '@/types/chat'

const TILE_URL =
  process.env.NEXT_PUBLIC_MAP_TILE_URL ||
  'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png'
const TILE_ATTRIBUTION =
  process.env.NEXT_PUBLIC_MAP_TILE_ATTRIBUTION ||
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'

const DATA_COLOR = '#2563EB'
const SELECTED_COLOR = '#111827'

interface LocationMapProps {
  coverage: SpatialCoverage | null
  point: MapPoint | null
  radiusKm: number
  onSelect: (point: MapPoint) => void
  height?: number
}

const FitToData: React.FC<{ coverage: SpatialCoverage | null }> = ({ coverage }) => {
  const map = useMap()
  useEffect(() => {
    const bounds = coverage?.bounds
    if (!bounds) {
      return
    }
    const box: LatLngBoundsExpression = [
      [bounds.min_lat, bounds.min_lon],
      [bounds.max_lat, bounds.max_lon],
    ]
    map.fitBounds(box, { padding: [24, 24] })
  }, [coverage, map])
  return null
}

const ClickToSelect: React.FC<{ onSelect: (point: MapPoint) => void }> = ({ onSelect }) => {
  useMapEvents({
    click: (event) => {
      // Leaflet longitudes can exceed ±180 after panning across the date line.
      const wrapped = event.latlng.wrap()
      onSelect({ latitude: wrapped.lat, longitude: wrapped.lng })
    },
  })
  return null
}

export const LocationMap: React.FC<LocationMapProps> = ({
  coverage,
  point,
  radiusKm,
  onSelect,
  height = 280,
}) => (
  <div style={{ height, width: '100%', borderRadius: 8, overflow: 'hidden' }}>
    <MapContainer
      center={[0, 0]}
      zoom={2}
      style={{ height: '100%', width: '100%', cursor: 'crosshair' }}
      scrollWheelZoom
    >
      <TileLayer url={TILE_URL} attribution={TILE_ATTRIBUTION} />
      <FitToData coverage={coverage} />
      <ClickToSelect onSelect={onSelect} />

      {coverage?.locations.map((location) => (
        <CircleMarker
          key={`${location.latitude},${location.longitude}`}
          center={[location.latitude, location.longitude]}
          radius={4}
          bubblingMouseEvents={false}
          pathOptions={{
            color: '#FFFFFF',
            weight: 1,
            fillColor: DATA_COLOR,
            fillOpacity: 0.85,
          }}
          eventHandlers={{
            click: () =>
              onSelect({ latitude: location.latitude, longitude: location.longitude }),
          }}
        >
          <Tooltip>
            {location.latitude.toFixed(4)}, {location.longitude.toFixed(4)} ·{' '}
            {location.simulations.toLocaleString()} simulations
          </Tooltip>
        </CircleMarker>
      ))}

      {point && (
        <>
          <Circle
            center={[point.latitude, point.longitude]}
            radius={radiusKm * 1000}
            pathOptions={{
              color: SELECTED_COLOR,
              weight: 1.5,
              dashArray: '4 4',
              fillColor: DATA_COLOR,
              fillOpacity: 0.08,
            }}
            interactive={false}
          />
          <CircleMarker
            center={[point.latitude, point.longitude]}
            radius={7}
            pathOptions={{
              color: '#FFFFFF',
              weight: 2,
              fillColor: SELECTED_COLOR,
              fillOpacity: 1,
            }}
          >
            <Tooltip permanent direction="top" offset={[0, -8]}>
              Selected point · {radiusKm} km
            </Tooltip>
          </CircleMarker>
        </>
      )}
    </MapContainer>
  </div>
)

export default LocationMap
