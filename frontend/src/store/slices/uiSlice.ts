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
    toggleSidebar: (state) => {
      state.sidebarOpen = !state.sidebarOpen;
    },
    showSnackbar: (state, action: PayloadAction<{ message: string; severity: UIState["snackbar"]["severity"] }>) => {
      state.snackbar.open = true;
      state.snackbar.message = action.payload.message;
      state.snackbar.severity = action.payload.severity;
    },
    hideSnackbar: (state) => {
      state.snackbar.open = false;
    },
  },
});

export const { toggleSidebar, showSnackbar, hideSnackbar } = uiSlice.actions;
export default uiSlice.reducer;