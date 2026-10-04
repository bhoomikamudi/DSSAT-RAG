/**
 * ChatContainer Component
 * Main chat interface container with message history and input
 */

import React, { useEffect, useRef, useState, useCallback } from 'react'
import {
  Box,
  Paper,
  Typography,
  Container,
  Alert,
  IconButton,
  Tooltip,
  Menu,
  MenuItem,
} from '@mui/material'
import {
  ChatMessage as ChatMessageType,
  ManagementVariables,
  MapPoint,
  SpatialCoverage,
} from '@/types/chat'
import { ChatMessage } from './ChatMessage'
import { ChatInput } from './ChatInput'
import { LocationFilterPanel } from './LocationFilterPanel'
import { ManagementVariablesPanel } from './ManagementVariablesPanel'
import chatbotApiService from '@/services/chatbotApi'
import { generateMessageId } from '@/utils/chatUtils'
import DeleteIcon from '@mui/icons-material/Delete'
import MoreVertIcon from '@mui/icons-material/MoreVert'
import FileDownloadIcon from '@mui/icons-material/FileDownload'
import SettingsIcon from '@mui/icons-material/Settings'

interface ChatContainerProps {
  title?: string
  subtitle?: string
}

export const ChatContainer: React.FC<ChatContainerProps> = ({
  title = 'DSSAT RAG Chatbot',
  subtitle = 'Ask questions about DSSAT agricultural data',
}) => {
  const [messages, setMessages] = useState<ChatMessageType[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [connectionError, setConnectionError] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null)
  // Lets the backend resolve follow-up analysis questions in this chat.
  const sessionIdRef = useRef<string>(generateMessageId())

  // Location filter: a map point and a radius. The default radius comes from
  // the backend configuration and the user can change it.
  const [mapPoint, setMapPoint] = useState<MapPoint | null>(null)
  const [radiusKm, setRadiusKm] = useState<number | null>(null)
  const [maxRadiusKm, setMaxRadiusKm] = useState<number>(500)
  const [coverage, setCoverage] = useState<SpatialCoverage | null>(null)
  const [coverageError, setCoverageError] = useState<string | null>(null)
  // Management fields and values present in the data (shown above the input).
  const [managementVariables, setManagementVariables] = useState<ManagementVariables | null>(null)
  const [managementError, setManagementError] = useState<string | null>(null)

  // Auto-scroll to bottom when new messages arrive
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Check API connectivity on mount
  useEffect(() => {
    const checkConnection = async () => {
      const isConnected = await chatbotApiService.healthCheck()
      setConnectionError(!isConnected)
    }
    checkConnection()
  }, [])

  // Load the default radius and where data exists (for the map).
  useEffect(() => {
    chatbotApiService
      .getSpatialConfig()
      .then((config) => {
        setRadiusKm((current) => current ?? config.default_radius_km)
        setMaxRadiusKm(config.max_radius_km)
      })
      .catch(() => undefined)
    chatbotApiService
      .getSpatialCoverage()
      .then(setCoverage)
      .catch(() => setCoverageError('Could not load data locations for the map.'))
    chatbotApiService
      .getManagementVariables()
      .then(setManagementVariables)
      .catch(() => setManagementError('Could not load the management variables.'))
  }, [])

  // Handle sending a message
  const handleSendMessage = useCallback(
    async (userQuery: string) => {
      try {
        // The same location values go into the request and onto the
        // question, so every question shows what its answer was filtered by.
        const location = { point: mapPoint, radiusKm }

        // Add user message
        const userMessage: ChatMessageType = {
          id: generateMessageId(),
          role: 'user',
          content: userQuery,
          timestamp: new Date(),
          sentLocation: location,
        }
        setMessages((prev) => [...prev, userMessage])

        // Add loading message for assistant response
        const loadingMessage: ChatMessageType = {
          id: generateMessageId(),
          role: 'assistant',
          content: '',
          timestamp: new Date(),
          isLoading: true,
        }
        setMessages((prev) => [...prev, loadingMessage])
        setIsLoading(true)

        // Send message to API
        const response = await chatbotApiService.sendMessage(
          userQuery,
          sessionIdRef.current,
          location
        )

        // Replace loading message with actual response
        setMessages((prev) => {
          const newMessages = [...prev]
          newMessages[newMessages.length - 1] = {
            id: generateMessageId(),
            role: 'assistant',
            content: response.answer,
            statistics: response.statistics,
            analysis: response.analysis,
            spatial: response.spatial,
            management: response.management,
            timestamp: new Date(),
          }
          return newMessages
        })

        // "Use all data" removed the location filter; clear the map selection.
        if (response.spatial?.cleared) {
          setMapPoint(null)
        }

        setConnectionError(false)
      } catch (err) {
        const errorMessage =
          err instanceof Error ? err.message : 'Failed to get response'

        // Replace loading message with error message
        setMessages((prev) => {
          const newMessages = [...prev]
          newMessages[newMessages.length - 1] = {
            id: generateMessageId(),
            role: 'assistant',
            content: '',
            timestamp: new Date(),
            error: errorMessage,
          }
          return newMessages
        })

        if (errorMessage.includes('No response from server')) {
          setConnectionError(true)
        }
      } finally {
        setIsLoading(false)
      }
    },
    [mapPoint, radiusKm]
  )

  const handleClearHistory = () => {
    setMessages([])
    sessionIdRef.current = generateMessageId()
    setAnchorEl(null)
  }

  const handleExportChat = () => {
    const chatContent = messages
      .map(
        (msg) =>
          `[${msg.role.toUpperCase()}]\n${msg.content}\n${msg.error ? `ERROR: ${msg.error}` : ''}\n`
      )
      .join('\n---\n\n')

    const element = document.createElement('a')
    element.setAttribute('href', `data:text/plain;charset=utf-8,${encodeURIComponent(chatContent)}`)
    element.setAttribute('download', `chat-export-${Date.now()}.txt`)
    element.style.display = 'none'
    document.body.appendChild(element)
    element.click()
    document.body.removeChild(element)
    setAnchorEl(null)
  }

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget)
  }

  const handleMenuClose = () => {
    setAnchorEl(null)
  }

  const hasMessages = messages.length > 0

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        height: '100vh',
        backgroundColor: '#F9FAFB',
      }}
    >
      {/* Header */}
      <Paper
        elevation={0}
        sx={{
          borderBottom: '1px solid #E5E7EB',
          backgroundColor: '#FFFFFF',
          p: 2.5,
        }}
      >
        <Container maxWidth="lg">
          <Box
            sx={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <Box>
              <Typography
                variant="h4"
                sx={{
                  fontWeight: 700,
                  color: '#111827',
                  mb: 0.5,
                }}
              >
                {title}
              </Typography>
              <Typography
                variant="body2"
                sx={{
                  color: '#6B7280',
                }}
              >
                {subtitle}
              </Typography>
            </Box>

            {/* Header Actions */}
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Tooltip title="Settings">
                <IconButton
                  size="small"
                  disabled
                  sx={{
                    color: '#9CA3AF',
                  }}
                >
                  <SettingsIcon />
                </IconButton>
              </Tooltip>
              <Tooltip title="More options">
                <IconButton
                  size="small"
                  onClick={handleMenuOpen}
                  sx={{
                    color: '#9CA3AF',
                  }}
                >
                  <MoreVertIcon />
                </IconButton>
              </Tooltip>
              <Menu
                anchorEl={anchorEl}
                open={Boolean(anchorEl)}
                onClose={handleMenuClose}
              >
                <MenuItem onClick={handleExportChat} disabled={!hasMessages}>
                  <FileDownloadIcon sx={{ mr: 1, fontSize: 18 }} />
                  Export Chat
                </MenuItem>
                <MenuItem onClick={handleClearHistory} disabled={!hasMessages}>
                  <DeleteIcon sx={{ mr: 1, fontSize: 18 }} />
                  Clear History
                </MenuItem>
              </Menu>
            </Box>
          </Box>
        </Container>
      </Paper>

      {/* Connection Error Alert */}
      {connectionError && (
        <Alert
          severity="error"
          sx={{
            m: 2,
            borderRadius: 2,
          }}
        >
          <Typography variant="body2" sx={{ fontWeight: 600, mb: 0.5 }}>
            Connection Error
          </Typography>
          <Typography variant="caption">
            Unable to connect to the DSSAT backend. Please ensure it is running on
            http://localhost:8005.
          </Typography>
        </Alert>
      )}

      {/* Messages Container */}
      <Box
        sx={{
          flex: 1,
          overflowY: 'auto',
          p: 2,
          display: 'flex',
          flexDirection: 'column',
          '&::-webkit-scrollbar': {
            width: '8px',
          },
          '&::-webkit-scrollbar-track': {
            backgroundColor: 'transparent',
          },
          '&::-webkit-scrollbar-thumb': {
            backgroundColor: '#D1D5DB',
            borderRadius: '4px',
            '&:hover': {
              backgroundColor: '#9CA3AF',
            },
          },
        }}
      >
        <Container maxWidth="lg">
          {!hasMessages ? (
            // Empty State
            <Box
              sx={{
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                alignItems: 'center',
                height: '100%',
                textAlign: 'center',
              }}
            >
              <Box
                sx={{
                  fontSize: 64,
                  mb: 2,
                }}
              >
                🌾
              </Box>
              <Typography
                variant="h5"
                sx={{
                  fontWeight: 700,
                  color: '#111827',
                  mb: 1,
                }}
              >
                Welcome to DSSAT RAG
              </Typography>
              <Typography
                variant="body2"
                sx={{
                  color: '#6B7280',
                  maxWidth: 400,
                  mb: 3,
                }}
              >
                Ask questions about DSSAT agricultural data. Start by typing a
                query below.
              </Typography>
              <Box
                sx={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                  gap: 2,
                  maxWidth: 600,
                  width: '100%',
                }}
              >
                {[
                  'What crop data do you have?',
                  'Tell me about maize cultivation',
                  'What nitrogen levels are available?',
                ].map((suggestion, idx) => (
                  <Paper
                    key={idx}
                    sx={{
                      p: 2,
                      cursor: 'pointer',
                      backgroundColor: '#FFFFFF',
                      border: '1px solid #E5E7EB',
                      borderRadius: 2,
                      transition: 'all 0.3s',
                      '&:hover': {
                        backgroundColor: '#F3F4F6',
                        borderColor: '#3B82F6',
                        boxShadow: '0 4px 12px rgba(30, 64, 175, 0.1)',
                      },
                    }}
                    onClick={() => handleSendMessage(suggestion)}
                  >
                    <Typography
                      variant="body2"
                      sx={{
                        color: '#111827',
                        fontSize: '0.9rem',
                      }}
                    >
                      {suggestion}
                    </Typography>
                  </Paper>
                ))}
              </Box>
            </Box>
          ) : (
            // Messages
            messages.map((message) => (
              <ChatMessage key={message.id} message={message} />
            ))
          )}
          <div ref={messagesEndRef} />
        </Container>
      </Box>

      {/* Management fields and values the data contains */}
      <ManagementVariablesPanel variables={managementVariables} error={managementError} />

      {/* Location filter (map point + radius) */}
      {radiusKm !== null && (
        <LocationFilterPanel
          point={mapPoint}
          radiusKm={radiusKm}
          maxRadiusKm={maxRadiusKm}
          coverage={coverage}
          coverageError={coverageError}
          onPointChange={setMapPoint}
          onRadiusChange={setRadiusKm}
        />
      )}

      {/* Input Container */}
      <ChatInput onSendMessage={handleSendMessage} isLoading={isLoading} />
    </Box>
  )
}

export default ChatContainer
