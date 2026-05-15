/* store/index.ts - Configuración del store de Redux
 *
 * Combina los slices (reducers) de la aplicación:
 * - analysis:        estado del análisis (grupos, sesión actual, carga)
 * - user:            datos del usuario autenticado
 * - ui:              estado de la interfaz (sidebar, snackbar)
 * - transcriptAudit: datos de auditoría de transcripciones
 *
 * Exporta los tipos RootState y AppDispatch para usar en hooks tipados.
 */

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