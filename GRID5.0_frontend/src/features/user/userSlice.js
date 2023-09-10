import { createSlice } from "@reduxjs/toolkit";

import {
  addUserToLocalStorage,
  getUserFromLocalStorage,
  removeUserFromLocalStorage,
} from "../../utils/localstorage";

const initialState = {
  user: getUserFromLocalStorage(),
};

const userSlice = createSlice({
  name: "user",
  initialState,
  reducers: {
    saveUser: (state, { payload }) => {
      const { token, user } = payload;
      state.user = { token, ...user };
      addUserToLocalStorage({ token, ...user });
    },
    logoutUser: (state) => {
      state.user = null;
      removeUserFromLocalStorage();
    },
  },
});

export const { logoutUser, saveUser } = userSlice.actions;
export default userSlice.reducer;
