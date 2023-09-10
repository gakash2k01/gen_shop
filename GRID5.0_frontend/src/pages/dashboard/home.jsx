import { useEffect, useState } from "react";
import { Container, Box } from "@mui/material";
import { ItemContainer, CircularLoader } from "../../components";
import { useSelector } from "react-redux";
import fetchDB, { fetchAI } from "../../utils/axios";
import authHeader from "../../utils/userAuthHeader";

const Home = () => {
  const [loading, setLoading] = useState(false);
  const [prevChats, setPrevChats] = useState([]);
  const [recomProducts, setRecomProducts] = useState([]);
  const { _id, token } = useSelector((state) => state.user.user);
  const fetchPrevChatsAndRecommendation = async () => {
    setLoading(true);
    const fetch1 = fetchDB.post(
      "/chat/getAll",
      { user_id: _id },
      authHeader(token)
    );
    const fetch2 = fetchDB.post(
      "/products/homeProducts",
      {},
      authHeader(token)
    );
    const [resp1, resp2] = await Promise.all([fetch1, fetch2]);
    const { chatFeed } = resp1.data;
    const recommendedItems = resp2.data;
    const cardDetails = chatFeed.map((el) => {
      return {
        title: el.title,
        timestamp: el.timeStamp,
        _id: el._id,
      };
    });
    console.log(recommendedItems);
    setRecomProducts(recommendedItems);
    setPrevChats(cardDetails);
    setLoading(false);
  };

  useEffect(() => {
    fetchPrevChatsAndRecommendation();
  }, []);

  if (loading) {
    return <CircularLoader />;
  }
  return (
    <Container maxWidth="ml" sx={{ flex: 1 }}>
      <ItemContainer title="Chats" ischats={true} data={prevChats} />
      <ItemContainer
        title="Products for you"
        ischats={false}
        showprice={true}
        data={recomProducts}
      />
    </Container>
  );
};

export default Home;
