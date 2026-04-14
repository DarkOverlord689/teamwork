import { configureStore } from "@reduxjs/toolkit";
import analysisSlice from "./slices/analysisSlice";
import userSlice from "./slices/userSlice";
import uiSlice from "./slices/uiSlice";
import transcriptAuditReducer from "./slices/transcriptAuditSlice";

export const store = configureStore({
  reducer: {
    analysis: analysisSlice,
    user: userSlice,
    ui: uiSlice,
    transcriptAudit: transcriptAuditReducer,
  },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;