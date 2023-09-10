import { Carousel } from "react-responsive-carousel";
import Box from "@mui/material/Box";
import "react-responsive-carousel/lib/styles/carousel.min.css"; // requires a loader

const data = [
  {
    image:
      "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0c/GoldenGateBridge-001.jpg/1200px-GoldenGateBridge-001.jpg",
  },
  {
    image:
      "https://cdn.britannica.com/s:800x450,c:crop/35/204435-138-2F2B745A/Time-lapse-hyper-lapse-Isle-Skye-Scotland.jpg",
  },
  {
    image:
      "https://static2.tripoto.com/media/filter/tst/img/735873/TripDocument/1537686560_1537686557954.jpg",
  },
  {
    image:
      "https://upload.wikimedia.org/wikipedia/commons/thumb/1/16/Palace_of_Fine_Arts_%2816794p%29.jpg/1200px-Palace_of_Fine_Arts_%2816794p%29.jpg",
  },
  {
    image:
      "https://i.natgeofe.com/n/f7732389-a045-402c-bf39-cb4eda39e786/scotland_travel_4x3.jpg",
  },
];

const CarouselItem = ({ data, handleThumbChange }) => {
  // const data = [{image: }]
  return (
    <div>
      <div>
        <Carousel onChange={handleThumbChange}>
          {data.map((el, idx) => (
            <Box key={idx} className="carousel-box">
              <img className="carousel-img" src={el.image} />
            </Box>
          ))}
        </Carousel>
      </div>
    </div>
  );
};

export default CarouselItem;
