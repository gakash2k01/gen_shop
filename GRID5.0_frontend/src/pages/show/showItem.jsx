import { Container, Box, Typography, ButtonGroup, Button } from "@mui/material";
import { CarouselItem } from "../../components";
import { CardPriceBox } from "../../components/itemCard/itemStyle";
import { percentOffCalc } from "../../utils/percentageCalculator";
import { AddIcon, RemoveIcon } from "../../icons";
import styled from "@emotion/styled";

const ModifiedButton = styled(Button)({
  borderColor: "#d2d1d1cb",
  borderRadius: "0",
  ":hover": {
    backgroundColor: "#e8e5e581",
    border: "none",
  },
});

const ShowItem = () => {
  return (
    <Container
      maxWidth="ml"
      sx={{
        display: "flex",
        gap: "3em",
      }}
    >
      <Box>
        <CarouselItem />
      </Box>
      <Box sx={{ display: "flex", flexDirection: "column", gap: "1em" }}>
        <Typography variant="h3">T-Shirt Name 10</Typography>
        <CardPriceBox fontSize="1.5em">
          <span className="price" style={{ fontSize: "1.2em" }}>
            ₹1399
          </span>
          <span className="actual-price" style={{ fontSize: "0.8em" }}>
            ₹3999
          </span>
          <span className="discount">{percentOffCalc(1399, 3999)}% off</span>
        </CardPriceBox>
        <Typography variant="body1">
          Lorem ipsum dolor, sit amet consectetur adipisicing elit. Excepturi
          quisquam soluta quod eligendi eaque qui sunt, laborum quidem nesciunt,
          inventore dolore? Quos repellendus quasi sequi iure voluptatibus porro
          in harum, unde rerum! Libero enim cum porro, assumenda vitae ipsam,
          veritatis consectetur, incidunt molestias accusamus praesentium velit!
          Velit exercitationem nam ad?
        </Typography>
        <ButtonGroup variant="outlined" aria-label="quantity">
          <ModifiedButton aria-label="delete">
            <AddIcon />
          </ModifiedButton>
          <ModifiedButton>1</ModifiedButton>
          <ModifiedButton aria-label="delete">
            <RemoveIcon />
          </ModifiedButton>
        </ButtonGroup>

        <Typography variant="h5">Description</Typography>
        <Box>
          <ol>
            <li>
              Lorem, ipsum dolor sit amet consectetur adipisicing elit. Ducimus
              soluta ut laboriosam animi dicta perspiciatis et, voluptates, esse
              hic nostrum, officiis consequatur? Iste, saepe accusantium.
            </li>
            <li>
              Lorem, ipsum dolor sit amet consectetur adipisicing elit. Ducimus
              soluta ut laboriosam animi dicta perspiciatis et, voluptates, esse
              hic nostrum, officiis consequatur? Iste, saepe accusantium.
            </li>
            <li>
              Lorem, ipsum dolor sit amet consectetur adipisicing elit. Ducimus
              soluta ut laboriosam animi dicta perspiciatis et, voluptates, esse
              hic nostrum, officiis consequatur? Iste, saepe accusantium.
            </li>
          </ol>
        </Box>
      </Box>
    </Container>
  );
};

export default ShowItem;
