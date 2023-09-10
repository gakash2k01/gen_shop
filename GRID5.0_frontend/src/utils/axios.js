import axios from "axios";

const fetchDB = axios.create({
  baseURL: "http://localhost:5000",
  headers: {
    "Content-Type": "application/json",
  },
});

export const fetchAI = axios.create({
  baseURL: "https://0167-34-71-113-122.ngrok.io",
  headers: {
    "Content-Type": "application/json",
  },
});

export default fetchDB;
