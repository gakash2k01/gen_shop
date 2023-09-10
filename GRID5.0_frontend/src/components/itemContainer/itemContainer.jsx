import { Box } from "@mui/material";
import Heading from "../heading/heading";
import ItemCard from "../itemCard/itemCard";
const ItemContainer = ({ title, data, ischats, showprice }) => {
  return (
    <Box sx={{ display: "flex", flexDirection: "column" }}>
      <Box
        sx={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <Heading siz="h2">{title}</Heading>
        {/* <span
          style={{ cursor: "pointer" }}
          onClick={() => navigate("category")}
        >
          View More
        </span> */}
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
        {data.map((el,idx) => (
          <ItemCard ischats={ischats} showprice={showprice} key={idx + el._id} {...el} />
        ))}
      </Box>
    </Box>
  );
};

export default ItemContainer;
