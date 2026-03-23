import { configureStore } from "@reduxjs/toolkit";
import analysisSlice from "./slices/analysisSlice";
import userSlice from "./slices/userSlice";
import uiSlice from "./slices/uiSlice";

export const store = configureStore({
  reducer: {
    analysis: analysisSlice,
    user: userSlice,
    ui: uiSlice,
  },
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;