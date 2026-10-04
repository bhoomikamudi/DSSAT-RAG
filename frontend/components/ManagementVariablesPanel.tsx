/**
 * ManagementVariablesPanel Component
 * Collapsible panel above the chat input listing the management fields and
 * values that exist in the ingested data, so users know what they can ask
 * about or filter on. Everything shown comes from the backend; nothing here
 * is hard-coded.
 */

import React, { useState } from 'react'
import {
  Box,
  Chip,
  Collapse,
  Container,
  IconButton,
  Paper,
  Tooltip,
  Typography,
} from '@mui/material'
import TuneIcon from '@mui/icons-material/Tune'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import ExpandLessIcon from '@mui/icons-material/ExpandLess'
import { ManagementVariables } from '@/types/chat'

interface ManagementVariablesPanelProps {
  variables: ManagementVariables | null
  error: string | null
}

export const ManagementVariablesPanel: React.FC<ManagementVariablesPanelProps> = ({
  variables,
  error,
}) => {
  const [open, setOpen] = useState(false)

  const summary = variables
    ? variables.fields
        .map((field) => `${field.values.length} ${field.label.toLowerCase()}${field.values.length === 1 ? '' : 's'}`)
        .join(' · ')
    : error ?? 'Loading…'

  return (
    <Paper elevation={0} sx={{ borderTop: '1px solid #E5E7EB', backgroundColor: '#FFFFFF' }}>
      <Container maxWidth="lg">
        <Box
          sx={{ display: 'flex', alignItems: 'center', gap: 1, py: 1, cursor: 'pointer' }}
          onClick={() => setOpen((value) => !value)}
        >
          <TuneIcon sx={{ fontSize: 20, color: '#2563EB' }} />
          <Typography variant="body2" sx={{ fontWeight: 600, color: '#111827' }}>
            Available management variables
          </Typography>
          <Chip
            size="small"
            label={summary}
            onClick={(event) => event.stopPropagation()}
            sx={{ maxWidth: '100%', backgroundColor: '#F3F4F6', color: '#374151' }}
          />
          <Box sx={{ flex: 1 }} />
          <Tooltip title={open ? 'Hide variables' : 'Show variables'}>
            <IconButton size="small" aria-label={open ? 'Hide variables' : 'Show variables'}>
              {open ? <ExpandLessIcon /> : <ExpandMoreIcon />}
            </IconButton>
          </Tooltip>
        </Box>

        <Collapse in={open} unmountOnExit>
          <Box sx={{ pb: 2, display: 'grid', gap: 1.5 }}>
            {error && (
              <Typography variant="caption" sx={{ color: '#DC2626' }}>
                {error}
              </Typography>
            )}
            {variables?.fields.map((field) => (
              <Box
                key={field.field}
                sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '130px 1fr' }, gap: 1, alignItems: 'start' }}
              >
                <Typography variant="body2" sx={{ color: '#374151', fontWeight: 600, pt: 0.25 }}>
                  {field.label}
                </Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75 }}>
                  {field.values.map((value) => (
                    <Tooltip
                      key={value.value}
                      title={`${value.simulations.toLocaleString()} simulations`}
                    >
                      <Chip
                        size="small"
                        variant="outlined"
                        label={value.label === value.value ? value.value : `${value.value} · ${value.label}`}
                        sx={{ borderColor: '#BFDBFE', color: '#1E3A8A', backgroundColor: '#EFF6FF' }}
                      />
                    </Tooltip>
                  ))}
                  {field.values.length === 1 && (
                    <Typography variant="caption" sx={{ color: '#6B7280', alignSelf: 'center' }}>
                      the only value in the data
                    </Typography>
                  )}
                </Box>
              </Box>
            ))}
            {variables && (
              <Typography variant="caption" sx={{ color: '#6B7280' }}>
                {variables.simulations.toLocaleString()} simulations
                {variables.crops.length ? ` of ${variables.crops.join(', ')}` : ''}
                {variables.years.min !== null ? `, ${variables.years.min}–${variables.years.max}` : ''}. Ask about
                one value (e.g. &ldquo;average yield for LNG planted 15 days late&rdquo;) or compare them (e.g.
                &ldquo;which cultivar has the highest average yield?&rdquo;). Without a filter, results combine
                all the values above.
              </Typography>
            )}
          </Box>
        </Collapse>
      </Container>
    </Paper>
  )
}

export default ManagementVariablesPanel
