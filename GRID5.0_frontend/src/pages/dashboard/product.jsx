import { Container, Box } from "@mui/material";
import { ItemCard, Heading } from "../../components";

const ProductPage = () => {
  return (
    <Container maxWidth="ml">
      <Heading siz="h2">Top Picks</Heading>
      <Box
        sx={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          my: "2em",
          "::after": {
            content: '""',
            flex: "auto",
          },
        }}
      >
        <ItemCard />
        <ItemCard />
        <ItemCard />
        <ItemCard />
        <ItemCard />
        <ItemCard />
        <ItemCard />
        <ItemCard />
        <ItemCard />
        <ItemCard />
        <ItemCard />
        <ItemCard />
        <ItemCard />
      </Box>
    </Container>
  );
};

export default ProductPage;
