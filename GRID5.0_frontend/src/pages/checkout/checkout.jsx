import { useEffect, useState } from "react";
import { Button } from "@mui/material";
import { CheckoutItem } from "../../components";
import { useDispatch, useSelector } from "react-redux";
import { clearCart } from "../../features/cart/cartSlice";
import toast from "react-hot-toast";

import "./style.css";
import fetchDB from "../../utils/axios";
import authHeader from "../../utils/userAuthHeader";

const CheckoutPage = () => {
  const dispatch = useDispatch();
  const { cart } = useSelector((state) => state.cart);
  const { token } = useSelector((state) => state.user.user);
  const [currCart, setCart] = useState([]);
  useEffect(() => {
    setCart(cart);
  }, [cart]);
  const totalCost = cart.reduce((total, curr) => {
    return total + curr.discountedPrice;
  }, 0);

  const handleBuy = async () => {
    try {
      const productsPurchased = cart.map((el) => el.id);
      const resp = await fetchDB.post(
        "/user/addPurchases",
        { productsPurchased },
        authHeader(token)
      );
      dispatch(clearCart());
    } catch (e) {
      toast.error("Something went wrong while placing order !");
      console.log(e);
    }
  };

  return (
    <div className="checkout-page">
      <div className="checkout-header">
        <div className="header-block">
          <span className="">Product</span>
        </div>
        <div className="header-block">
          <span className="">Name</span>
        </div>
        <div className="header-block">
          <span className="">Price</span>
        </div>
        <div className="header-block">
          <span className="">Discounted Price</span>
        </div>
        <div className="header-block">
          <span className="">Remove</span>
        </div>
      </div>
      {currCart.map((el, idx) => (
        <CheckoutItem
          key={idx}
          idx={idx}
          name={el.name}
          price={el.price}
          imageUrl={el.image}
          discountedPrice={el.discountedPrice}
        />
      ))}
      <div className="total">
        <span>Total: ₹{totalCost}</span>
      </div>
      <Button
        sx={{
          backgroundColor: "#acababcd",
          display: "block",
        }}
        onClick={handleBuy}
      >
        Pay
      </Button>
    </div>
  );
};

export default CheckoutPage;
