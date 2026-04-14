import { createAsyncThunk, createSlice } from '@reduxjs/toolkit';

import type {
  AudioResult,
  TranscriptSegment,
  VisionResult,
} from '../../types/audit';
import { audioService, visionService } from '../../services/api';

interface SessionAuditData {
  transcripts: TranscriptSegment[];
  audioResult: AudioResult | null;
  visionResult: Omit<VisionResult, 'person_embeddings'> | null;
  loadingTranscripts: boolean;
  loadingAudio: boolean;
  loadingVision: boolean;
  errorTranscripts: string | null;
  errorAudio: string | null;
  errorVision: string | null;
}

interface TranscriptAuditState {
  sessions: Record<string, SessionAuditData>;
}

function emptySession(): SessionAuditData {
  return {
    transcripts: [],
    audioResult: null,
    visionResult: null,
    loadingTranscripts: false,
    loadingAudio: false,
    loadingVision: false,
    errorTranscripts: null,
    errorAudio: null,
    errorVision: null,
  };
}

const initialState: TranscriptAuditState = {
  sessions: {},
};

export const fetchTranscripts = createAsyncThunk(
  'transcriptAudit/fetchTranscripts',
  async (sessionId: string) => {
    const data = await audioService.getTranscripts(sessionId);
    return { sessionId, data };
  },
);

export const fetchAudioResult = createAsyncThunk(
  'transcriptAudit/fetchAudioResult',
  async (sessionId: string) => {
    const data = await audioService.getResults(sessionId);
    return { sessionId, data };
  },
);

export const fetchVisionResult = createAsyncThunk(
  'transcriptAudit/fetchVisionResult',
  async (sessionId: string) => {
    const data = await visionService.getResults(sessionId);
    const { person_embeddings: _, ...rest } = data;
    return { sessionId, data: rest };
  },
);

const transcriptAuditSlice = createSlice({
  name: 'transcriptAudit',
  initialState,
  reducers: {
    clearSession(state, action: { payload: string }) {
      delete state.sessions[action.payload];
    },
  },
  extraReducers: (builder) => {
    // fetchTranscripts
    builder
      .addCase(fetchTranscripts.pending, (state, action) => {
        const id = action.meta.arg;
        if (!state.sessions[id]) state.sessions[id] = emptySession();
        state.sessions[id].loadingTranscripts = true;
        state.sessions[id].errorTranscripts = null;
      })
      .addCase(fetchTranscripts.fulfilled, (state, action) => {
        const { sessionId, data } = action.payload;
        if (!state.sessions[sessionId]) state.sessions[sessionId] = emptySession();
        state.sessions[sessionId].transcripts = data;
        state.sessions[sessionId].loadingTranscripts = false;
      })
      .addCase(fetchTranscripts.rejected, (state, action) => {
        const id = action.meta.arg;
        if (!state.sessions[id]) state.sessions[id] = emptySession();
        state.sessions[id].loadingTranscripts = false;
        state.sessions[id].errorTranscripts = action.error.message ?? 'Unknown error';
      });

    // fetchAudioResult
    builder
      .addCase(fetchAudioResult.pending, (state, action) => {
        const id = action.meta.arg;
        if (!state.sessions[id]) state.sessions[id] = emptySession();
        state.sessions[id].loadingAudio = true;
        state.sessions[id].errorAudio = null;
      })
      .addCase(fetchAudioResult.fulfilled, (state, action) => {
        const { sessionId, data } = action.payload;
        if (!state.sessions[sessionId]) state.sessions[sessionId] = emptySession();
        state.sessions[sessionId].audioResult = data;
        state.sessions[sessionId].loadingAudio = false;
      })
      .addCase(fetchAudioResult.rejected, (state, action) => {
        const id = action.meta.arg;
        if (!state.sessions[id]) state.sessions[id] = emptySession();
        state.sessions[id].loadingAudio = false;
        state.sessions[id].errorAudio = action.error.message ?? 'Unknown error';
      });

    // fetchVisionResult
    builder
      .addCase(fetchVisionResult.pending, (state, action) => {
        const id = action.meta.arg;
        if (!state.sessions[id]) state.sessions[id] = emptySession();
        state.sessions[id].loadingVision = true;
        state.sessions[id].errorVision = null;
      })
      .addCase(fetchVisionResult.fulfilled, (state, action) => {
        const { sessionId, data } = action.payload;
        if (!state.sessions[sessionId]) state.sessions[sessionId] = emptySession();
        state.sessions[sessionId].visionResult = data;
        state.sessions[sessionId].loadingVision = false;
      })
      .addCase(fetchVisionResult.rejected, (state, action) => {
        const id = action.meta.arg;
        if (!state.sessions[id]) state.sessions[id] = emptySession();
        state.sessions[id].loadingVision = false;
        state.sessions[id].errorVision = action.error.message ?? 'Unknown error';
      });
  },
});

export const { clearSession } = transcriptAuditSlice.actions;
export default transcriptAuditSlice.reducer;
