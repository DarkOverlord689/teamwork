import { useEffect, useState } from 'react';
import {
  Alert, Accordion, AccordionSummary, AccordionDetails,
  Box, Chip, LinearProgress, Table,
  TableBody, TableCell, TableHead, TableRow, TextField, Typography, Button
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { useDispatch, useSelector } from 'react-redux';
import type { RootState, AppDispatch } from '../../store';
import { fetchTranscripts } from '../../store/slices/transcriptAuditSlice';
import { getSpeakerColor } from './utils/speakerColor';
import { formatTime } from './utils/timeFormat';
import type { TranscriptSegment } from '../../types/audit';

export default function TranscriptView({ sessionId }: { sessionId: string }) {
  const dispatch = useDispatch<AppDispatch>();
  const session = useSelector((state: RootState) =>
    state.transcriptAudit.sessions[sessionId]
  );

  const transcripts = session?.transcripts ?? [];
  const loading = session?.loadingTranscripts ?? false;
  const error = session?.errorTranscripts ?? null;

  const [searchQuery, setSearchQuery] = useState('');
  const [activeSpeakers, setActiveSpeakers] = useState<Set<string>>(new Set());
  const [retryEnabled, setRetryEnabled] = useState(false);

  useEffect(() => {
    // Only fetch if there is no session entry yet (never fetched for this sessionId).
    // Checking !session avoids re-fetching when transcripts legitimately came back empty.
    if (!session) {
      dispatch(fetchTranscripts(sessionId));
    }
  }, [sessionId]);

  // Initialize active speakers when transcripts load
  useEffect(() => {
    const ids = new Set(transcripts.map((t: TranscriptSegment) => t.speaker_id));
    setActiveSpeakers(ids);
  }, [transcripts.length]);

  const uniqueSpeakers = Array.from(new Set(transcripts.map((t: TranscriptSegment) => t.speaker_id)));

  const toggleSpeaker = (id: string) => {
    setActiveSpeakers(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  const filtered = transcripts.filter((seg: TranscriptSegment) => {
    const q = searchQuery.toLowerCase();
    const matchesText = q === '' || seg.text.toLowerCase().includes(q) || seg.speaker_id.toLowerCase().startsWith(q);
    const matchesSpeaker = activeSpeakers.size === 0 || activeSpeakers.has(seg.speaker_id);
    return matchesText && matchesSpeaker;
  });

  if (loading) return <LinearProgress />;

  if (error) {
    const is409 = error.includes('409');
    return (
      <Alert severity="error" sx={{ m: 2 }}
        action={
          <Button size="small" disabled={is409 && !retryEnabled}
            onClick={() => { setRetryEnabled(false); dispatch(fetchTranscripts(sessionId)); }}>
            Reintentar
          </Button>
        }>
        {is409
          ? 'El análisis aún no ha finalizado. Intente nuevamente en unos momentos.'
          : 'Error al cargar la transcripción.'}
      </Alert>
    );
  }

  if (transcripts.length === 0) {
    return (
      <Box sx={{ p: 3, textAlign: 'center' }}>
        <Typography color="text.secondary">No hay transcripción disponible para esta sesión.</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 2 }}>
      <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', mb: 2, alignItems: 'center' }}>
        <TextField
          size="small"
          placeholder="Buscar por texto o speaker..."
          value={searchQuery}
          onChange={e => setSearchQuery(e.target.value)}
          sx={{ minWidth: 260 }}
        />
        {uniqueSpeakers.map(id => (
          <Chip
            key={id}
            label={id}
            onClick={() => toggleSpeaker(id)}
            sx={{
              bgcolor: activeSpeakers.has(id) ? getSpeakerColor(id) : '#e0e0e0',
              color: activeSpeakers.has(id) ? 'white' : 'text.secondary',
              fontWeight: 500,
            }}
          />
        ))}
      </Box>

      {filtered.length === 0 && (
        <Typography color="text.secondary" sx={{ p: 2 }}>
          Sin resultados para la búsqueda.
        </Typography>
      )}

      {filtered.map((seg: TranscriptSegment, i: number) => {
        const hasWords = seg.words && seg.words.length > 0;
        const lowConfidence = seg.no_speech_prob > 0.8;
        const preview = seg.text.slice(0, 80) + (seg.text.length > 80 ? '…' : '');
        return (
          <Accordion key={`${seg.start}-${i}`} disableGutters>
            <AccordionSummary expandIcon={hasWords ? <ExpandMoreIcon /> : null}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap', width: '100%' }}>
                <Chip
                  size="small"
                  label={seg.speaker_id}
                  sx={{ bgcolor: getSpeakerColor(seg.speaker_id), color: 'white', fontWeight: 600 }}
                />
                <Typography variant="caption" color="text.secondary">
                  {formatTime(seg.start)} → {formatTime(seg.end)}
                </Typography>
                <Typography variant="body2" sx={{ flex: 1 }}>{preview}</Typography>
                {lowConfidence && (
                  <Typography variant="caption" color="text.disabled">(baja confianza)</Typography>
                )}
                {!hasWords && (
                  <Typography variant="caption" color="text.disabled">Sin detalle por palabra</Typography>
                )}
              </Box>
            </AccordionSummary>
            {hasWords && (
              <AccordionDetails>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Palabra</TableCell>
                      <TableCell>Inicio (s)</TableCell>
                      <TableCell>Fin (s)</TableCell>
                      <TableCell>Confianza (%)</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {seg.words.map((w, wi) => (
                      <TableRow key={wi}>
                        <TableCell>{w.word}</TableCell>
                        <TableCell>{w.start.toFixed(2)}</TableCell>
                        <TableCell>{w.end.toFixed(2)}</TableCell>
                        <TableCell>{(w.confidence * 100).toFixed(0)}%</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </AccordionDetails>
            )}
          </Accordion>
        );
      })}
    </Box>
  );
}
