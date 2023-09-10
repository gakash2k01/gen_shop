import Card from "@mui/material/Card";
import { Box } from "@mui/material";
import CardContent from "@mui/material/CardContent";
import CardMedia from "@mui/material/CardMedia";
import Typography from "@mui/material/Typography";
import { useDispatch } from "react-redux";
import { useNavigate } from "react-router-dom";
import { addToCart } from "../../features/cart/cartSlice";
import moment from "moment";

import { CardPriceBox } from "./itemStyle";
import { percentOffCalc } from "../../utils/percentageCalculator";

const default_name = "Wedding outfit for Men";

const ItemCard = ({
  imgUrl,
  title,
  ischats,
  _id,
  showprice,
  price,
  discountedPrice,
  timestamp,
}) => {
  const navigate = useNavigate();
  const dispatch = useDispatch();
  const handleCardClick = async () => {
    if (ischats) {
      navigate("/chat", { state: { chatbox_id: _id } });
    } else {
      dispatch(
        addToCart({
          item: {
            image: imgUrl,
            name: title,
            id: _id,
            price,
            discountedPrice,
          },
        })
      );
    }
  };
  return (
    <Card
      className="item-card"
      sx={{
        maxWidth: "400px",
        width: "250px",
        cursor: "pointer",
        my: "1em",
        backgroundColor: ischats?"#e0ecf5":"",
      }}
      onClick={handleCardClick}
    >
      {imgUrl && (
        <CardMedia
          image={imgUrl}
          sx={{
            mixBlendMode: "multiply",
            backgroundSize: "cover",
            backgroundRepeat: "no-repeat",
            backgroundPosition: "center",
            height: 300,
            width: "100%",
            my: "1em",
          }}
        />
      )}

      <CardContent sx={{ pt: "0.5em", pb: "0.5em !important", px: "1em" }}>
        <Typography
          gutterBottom
          variant="overline"
          component="div"
          className="truncate card-title"
          sx={{ lineHeight: "20px" }}
        >
          {title || default_name}
        </Typography>
        {showprice && (
          <Box sx={{ display: "flex" }}>
            <CardPriceBox>
              <span className="price">₹{discountedPrice}</span>
              <span className="actual-price">₹{price}</span>
              <span className="discount">
                {percentOffCalc(discountedPrice, price)}% off
              </span>
            </CardPriceBox>
          </Box>
        )}
        {ischats && (
          <Box sx={{ display: "flex" }}>
            <CardPriceBox>
              <span className="price">
                Created at:{" "}
                {moment(timestamp).format("MMMM Do YYYY, h:mm:ss a")}
              </span>
            </CardPriceBox>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default ItemCard;
