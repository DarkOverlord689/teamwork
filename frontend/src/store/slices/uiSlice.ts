/* uiSlice.ts - Estado global de la interfaz de usuario
 *
 * Controla elementos visuales compartidos en toda la app:
 * - sidebarOpen: indica si el menú lateral está abierto o cerrado
 * - snackbar:    mensajes temporales que se muestran al usuario
 *                (éxito, error, advertencia o información)
 */

import { createSlice, PayloadAction } from "@reduxjs/toolkit";

interface UIState {
  sidebarOpen: boolean;
  snackbar: {
    open: boolean;
    message: string;
    severity: "success" | "error" | "warning" | "info";
  };
}

const initialState: UIState = {
  sidebarOpen: true,
  snackbar: {
    open: false,
    message: "",
    severity: "info",
  },
};

const uiSlice = createSlice({
  name: "ui",
  initialState,
  reducers: {
    /* Abre o cierra el menú lateral */
    toggleSidebar: (state) => {
      state.sidebarOpen = !state.sidebarOpen;
    },
    /* Muestra un mensaje en el Snackbar */
    showSnackbar: (state, action: PayloadAction<{ message: string; severity: UIState["snackbar"]["severity"] }>) => {
      state.snackbar.open = true;
      state.snackbar.message = action.payload.message;
      state.snackbar.severity = action.payload.severity;
    },
    /* Oculta el Snackbar */
    hideSnackbar: (state) => {
      state.snackbar.open = false;
    },
  },
});

export const { toggleSidebar, showSnackbar, hideSnackbar } = uiSlice.actions;
export default uiSlice.reducer;