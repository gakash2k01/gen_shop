import { useState, useEffect } from "react";
import { Box, Button, Container } from "@mui/material";
import { useLocation } from "react-router-dom";
import { useSelector } from "react-redux";
import Heading from "../../components/heading/heading";
import { ItemCard, CircularLoader } from "../../components";
import authHeader from "../../utils/userAuthHeader";
import fetchDB from "../../utils/axios";

const SuggPage = () => {
  const { state } = useLocation();
  const [loading, setLoading] = useState(false);
  const { token } = useSelector((state) => state.user.user);
  const [recommendedProducts, setRecommendedProducts] = useState([]);
  const [encoding, setEncoding] = useState(state?.imageEncoding);

  const fetchRecommendations = async () => {
    try {
      setLoading(true);
      // console.log(encoding);
      const resp = await fetchDB.post(
        "/products/pics",
        { imageEncoding: encoding },
        authHeader(token)
      );
      setRecommendedProducts(resp.data);
      setLoading(false);
    } catch (e) {
      console.log(e);
    }
  };
  useEffect(() => {
    fetchRecommendations();
  }, []);


  if (loading) {
    return <CircularLoader />;
  }

  return (
    <Container maxWidth="ml">
      <Box sx={{ display: "flex", flexDirection: "column" }}>
        <Box
          sx={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <Heading siz="h2">Related Products</Heading>
          <span
            style={{ cursor: "pointer" }}
            //   onClick={() => navigate("category")}
          >
            View More
          </span>
        </Box>
        <Box
          sx={{
            display: "flex",
            justifyContent: "space-around",
            flexWrap: "wrap",
            gap: "2em",
            my: "2em",
          }}
        >
          {recommendedProducts.map((el) => (
            <ItemCard
              key={el.productId}
              title={el.productName}
              price={el.productPrice}
              discountedPrice={el.discountedPrice}
              imgUrl={"data:image/png;base64," + el.imageEncoding}
              _id={el.productId}
              showprice
            />
          ))}
        </Box>
      </Box>
    </Container>
  );
};

export default SuggPage;
