import { createSlice, PayloadAction } from "@reduxjs/toolkit";

interface Group {
  id: string;
  name: string;
  course_id: string;
  created_at: string;
}

interface Session {
  id: string;
  group_id: string;
  status: "pending" | "processing" | "completed" | "error";
  video_path: string;
  duration_seconds: number;
  processed_at: string;
}

interface AnalysisState {
  groups: Group[];
  currentSession: Session | null;
  loading: boolean;
  error: string | null;
}

const initialState: AnalysisState = {
  groups: [],
  currentSession: null,
  loading: false,
  error: null,
};

const analysisSlice = createSlice({
  name: "analysis",
  initialState,
  reducers: {
    setGroups: (state, action: PayloadAction<Group[]>) => {
      state.groups = action.payload;
    },
    setCurrentSession: (state, action: PayloadAction<Session | null>) => {
      state.currentSession = action.payload;
    },
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.loading = action.payload;
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
  },
});

export const { setGroups, setCurrentSession, setLoading, setError } = analysisSlice.actions;
export default analysisSlice.reducer;