/**
 * LocationFilterPanel Component
 * Collapsible panel above the chat input: pick a point on the map and set
 * the radius used for location-based questions.
 */

import React, { useState } from 'react'
import dynamic from 'next/dynamic'
import {
  Box,
  Button,
  Chip,
  Collapse,
  Container,
  IconButton,
  Paper,
  Slider,
  TextField,
  Tooltip,
  Typography,
} from '@mui/material'
import PlaceIcon from '@mui/icons-material/Place'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import ExpandLessIcon from '@mui/icons-material/ExpandLess'
import { MapPoint, SpatialCoverage } from '@/types/chat'

// Leaflet touches `window`, so the map renders only in the browser.
const LocationMap = dynamic(() => import('./LocationMap'), {
  ssr: false,
  loading: () => (
    <Box sx={{ height: 280, display: 'grid', placeItems: 'center', color: '#6B7280' }}>
      <Typography variant="caption">Loading map…</Typography>
    </Box>
  ),
})

interface LocationFilterPanelProps {
  point: MapPoint | null
  radiusKm: number
  maxRadiusKm: number
  coverage: SpatialCoverage | null
  coverageError: string | null
  onPointChange: (point: MapPoint | null) => void
  onRadiusChange: (radiusKm: number) => void
}

export const LocationFilterPanel: React.FC<LocationFilterPanelProps> = ({
  point,
  radiusKm,
  maxRadiusKm,
  coverage,
  coverageError,
  onPointChange,
  onRadiusChange,
}) => {
  const [open, setOpen] = useState(false)
  const [radiusText, setRadiusText] = useState<string | null>(null)

  const sliderMax = Math.min(maxRadiusKm, 200)

  const commitRadius = (value: number) => {
    if (Number.isFinite(value) && value > 0 && value <= maxRadiusKm) {
      onRadiusChange(Math.round(value * 10) / 10)
    }
    setRadiusText(null)
  }

  const summary = point
    ? `Within ${radiusKm} km of ${point.latitude.toFixed(4)}, ${point.longitude.toFixed(4)}`
    : `No point selected · radius ${radiusKm} km for places named in questions`

  return (
    <Paper
      elevation={0}
      sx={{ borderTop: '1px solid #E5E7EB', backgroundColor: '#FFFFFF' }}
    >
      <Container maxWidth="lg">
        <Box
          sx={{ display: 'flex', alignItems: 'center', gap: 1, py: 1, cursor: 'pointer' }}
          onClick={() => setOpen((value) => !value)}
        >
          <PlaceIcon sx={{ fontSize: 20, color: point ? '#2563EB' : '#9CA3AF' }} />
          <Typography variant="body2" sx={{ fontWeight: 600, color: '#111827' }}>
            Location filter
          </Typography>
          <Chip
            size="small"
            label={summary}
            onDelete={point ? () => onPointChange(null) : undefined}
            onClick={(event) => event.stopPropagation()}
            sx={{
              maxWidth: '100%',
              backgroundColor: point ? '#EFF6FF' : '#F3F4F6',
              color: '#374151',
            }}
          />
          <Box sx={{ flex: 1 }} />
          <Tooltip title={open ? 'Hide map' : 'Show map'}>
            <IconButton size="small" aria-label={open ? 'Hide map' : 'Show map'}>
              {open ? <ExpandLessIcon /> : <ExpandMoreIcon />}
            </IconButton>
          </Tooltip>
        </Box>

        <Collapse in={open} unmountOnExit>
          <Box sx={{ pb: 2 }}>
            <Typography variant="caption" sx={{ display: 'block', color: '#6B7280', mb: 1 }}>
              Click the map (or a blue data location) to filter answers to simulations
              within the radius. Questions naming a place, such as &ldquo;near Kitale&rdquo;, use
              the same radius unless the question gives its own.
            </Typography>

            {coverageError ? (
              <Typography variant="caption" sx={{ color: '#DC2626' }}>
                {coverageError}
              </Typography>
            ) : (
              <LocationMap
                coverage={coverage}
                point={point}
                radiusKm={radiusKm}
                onSelect={onPointChange}
              />
            )}

            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                gap: 2,
                mt: 1.5,
                flexWrap: 'wrap',
              }}
            >
              <Typography variant="body2" sx={{ color: '#374151', minWidth: 60 }}>
                Radius
              </Typography>
              <Slider
                value={Math.min(radiusKm, sliderMax)}
                min={1}
                max={sliderMax}
                step={1}
                onChange={(_, value) => onRadiusChange(value as number)}
                sx={{ flex: 1, minWidth: 160 }}
                aria-label="Radius in kilometres"
              />
              <TextField
                size="small"
                type="number"
                value={radiusText ?? String(radiusKm)}
                onChange={(event) => setRadiusText(event.target.value)}
                onBlur={() => commitRadius(Number(radiusText ?? radiusKm))}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') {
                    commitRadius(Number(radiusText ?? radiusKm))
                  }
                }}
                inputProps={{ min: 1, max: maxRadiusKm, step: 1, 'aria-label': 'Radius (km)' }}
                InputProps={{ endAdornment: <Typography variant="caption">km</Typography> }}
                sx={{ width: 110 }}
              />
              <Button
                size="small"
                variant="outlined"
                disabled={!point}
                onClick={() => onPointChange(null)}
              >
                Clear point
              </Button>
            </Box>

            {coverage && (
              <Typography variant="caption" sx={{ display: 'block', color: '#6B7280', mt: 1 }}>
                Data covers {coverage.location_count.toLocaleString()} locations
                {coverage.locations_truncated ? ' (map shows the first 5,000)' : ''} ·{' '}
                {coverage.simulations.toLocaleString()} simulations
              </Typography>
            )}
          </Box>
        </Collapse>
      </Container>
    </Paper>
  )
}

export default LocationFilterPanel
