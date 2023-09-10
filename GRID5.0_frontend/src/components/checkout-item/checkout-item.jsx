import "./checkout-item.style.css";
import { removeFromCart } from "../../features/cart/cartSlice";
import { useDispatch } from "react-redux";

const CheckoutItem = ({ name, price, imageUrl, discountedPrice, idx }) => {
  const dispatch = useDispatch();
  return (
    <div className="checkout-item">
      <div className="image-container">
        <img src={imageUrl} alt="item" />
      </div>
      <span className="name">{name}</span>
      <span className="price">{price}</span>
      <span className="discounted price">{discountedPrice}</span>
      <div
        className="remove-button"
        onClick={() => {
          console.log("removing, ", idx);
          dispatch(removeFromCart(idx));
        }}
      >
        &#10005;
      </div>
    </div>
  );
};

export default CheckoutItem;
