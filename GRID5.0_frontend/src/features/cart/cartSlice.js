import { createSlice } from "@reduxjs/toolkit";
import toast from "react-hot-toast";

const initialState = {
  cart: [],
};

const cartSlice = createSlice({
  name: "cart",
  initialState,
  reducers: {
    addToCart: (state, { payload }) => {
      const { item } = payload;
      console.log("slice = ", item);
      // {endoding: ,name:, price:,discountedprice:}
      state.cart.push(item);
    },
    removeFromCart: (state, { payload }) => {
      const { idx } = payload;
      state.cart.splice(idx, 1);
    },
    clearCart: (state) => {
        state.cart = [];
        toast.success("Order successful !");
    }
  },
});

export const { addToCart, removeFromCart, clearCart } = cartSlice.actions;
export default cartSlice.reducer;
