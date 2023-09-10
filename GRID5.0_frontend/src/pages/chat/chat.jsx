import { useEffect, useState } from "react";
import { toast } from "react-hot-toast";
import { Box, TextField, Button, ButtonGroup } from "@mui/material";
import fetchDB from "../../utils/axios";
import { useNavigate, useLocation } from "react-router-dom";
import { fetchAI } from "../../utils/axios";
import {
  CarouselItem,
  AIMessageBox,
  ReceiverMessageBox,
  NoImage,
} from "../../components";
import { ChatIcon } from "../../icons";
import { ChatBox } from "./style";
import { useSelector } from "react-redux";
import authHeader from "../../utils/userAuthHeader";

const ChatAI = () => {
  const navigate = useNavigate();

  /* state variables */
  const { state } = useLocation();
  const { token } = useSelector((state) => state.user.user);
  const [chatbox_id, setChatbox_id] = useState(state?.chatbox_id);
  const [text, setText] = useState({ sender: 0, message: "", timestamp: "" });
  const [_chats, setChats] = useState([]);
  const [_summary, setSummary] = useState("");
  const [image, setImage] = useState("");
  const [selectedImg, setSelectedImg] = useState("");

  /* Helper Functions */

  const updateText = (e) => {
    setText((s) => ({ ...s, message: e.target.value, timestamp: new Date() }));
  };

  const fetchData = async () => {
    try {
      const resp = await fetchDB.post(
        "/chat/get",
        { chatBoxId: chatbox_id },
        authHeader(token)
      );
      const { chats, images, summary } = resp.data;
      // console.log(chats);
      setChats(chats);
      setImage([{ image: images }]);
      setSummary(summary);
    } catch (e) {
      toast.error("Something went wrong while fetching chats");
      console.log(e);
    }
  };

  const handleSend = async () => {
    const ttext = text;
    // add prev conversations
    const prev_conv = _chats.slice(-6).map((el) => el.message);
    setChats([
      ..._chats,
      { sender: 0, message: text.message, timestamp: text.timestamp },
      { sender: 1, message: "typing...", timestamp: new Date() },
    ]);
    setText((s) => ({ ...s, message: "" }));

    text.timestamp = new Date();
    const data = [];
    /*pushing summary , prev_conv, prompt */
    data.push(_summary);
    data.push(...prev_conv);
    data.push(ttext.message);

    // console.log(data);
    const ai_resp = await fetchAI.post("/lang_model", { inp: data });
    const [newSummary, ai_txt_resp] = ai_resp.data.split("!?!?");
    // console.log("ai resp = ", ai_txt_resp);
    // send user text and response msg to backend
    const ai_txt = { sender: 1, message: ai_txt_resp, timestamp: new Date() };
    const ssdata = {
      ai_txt,
      user_txt: ttext,
      summary: newSummary,
      chatBoxId: chatbox_id,
    };
    const resp = await fetchDB.post("/chat/addChat", ssdata, authHeader(token));
    const { chats, images, summary, _id } = resp.data;
    // console.log(chats);
    setChatbox_id(_id);
    setChats(chats);
    setImage([{ image: images }]);
    setSummary(summary);
  };

  const handleUpdate = async () => {
    try {
      setChats([
        ..._chats,
        { sender: 0, message: text.message, timestamp: text.timestamp },
        { sender: 1, message: "updating image...", timestamp: new Date() },
      ]);
      const data = {
        inp: [text.message, selectedImg],
      };
      setText((s) => ({ ...s, message: "" }));
      const resp = await fetchAI.post("/image_model", data);
      setImage([{ image: "data:image/png;base64," + resp.data }]);

      const newChat = _chats;
      newChat.pop();
      
    } catch (e) {
      console.log(e);
    }
  };

  const handleGenerate = async () => {
    try {
      const ttext = text;
      setChats([
        ..._chats,
        { sender: 0, message: text.message, timestamp: text.timestamp },
        { sender: 1, message: "generating...", timestamp: new Date() },
      ]);
      setText((s) => ({ ...s, message: "" }));
      const ssdata = {
        user_txt: ttext,
        chatBoxId: chatbox_id,
      };
      const resp = await fetchDB.post(
        "/chat/generate",
        ssdata,
        authHeader(token)
      );
      // console.log(resp.data);
      const { chatBox, imageList } = resp.data;
      const { chats, _id } = chatBox;
      setChatbox_id(_id);
      setChats(chats);
      const updatedImgList = imageList.map((imgBase64) => {
        return {
          image: "data:image/png;base64," + imgBase64,
        };
      });
      setSelectedImg(updatedImgList[0].image.slice(22));
      setImage(updatedImgList);
    } catch (e) {
      console.log(e);
    }
  };

  const handlePick = async () => {
    try {
      navigate("/recommended", { state: { imageEncoding: selectedImg } });
    } catch (e) {
      console.log(e);
    }
  };

  const handleThumbChange = (e) => {
    setSelectedImg(image[e].image.slice(22));
  };

  /* useEffect hooks */

  useEffect(() => {
    if (chatbox_id) fetchData();
  }, []);

  return (
    <ChatBox maxWidth="ml" sx={{ display: "flex", gap: "2em", height: "80vh" }}>
      <Box className="left-box">
        {image.length && image[0].image ? (
          <>
            <CarouselItem data={image} handleThumbChange={handleThumbChange} />
            <Button
              sx={{
                backgroundColor: "#acababcd",
                display: "block",
                mx: "auto",
              }}
              onClick={handlePick}
            >
              Pick this product
            </Button>
          </>
        ) : (
          <NoImage />
        )}
      </Box>
      <Box className="right-box">
        {_chats.length ? (
          <Box className="chat-box">
            {_chats.map((obj, idx) => {
              if (obj.sender)
                return (
                  <AIMessageBox
                    key={idx}
                    text={obj.message}
                    timestamp={obj.timestamp}
                  />
                );
              else {
                return (
                  <ReceiverMessageBox
                    key={idx}
                    text={obj.message}
                    timestamp={obj.timestamp}
                  />
                );
              }
            })}
          </Box>
        ) : (
          <ChatIcon sx={{ fontSize: 140, color: "#9f9d9d", my: "auto" }} />
        )}
        <Box className="message-input">
          <TextField
            // multiline
            maxRows={4}
            placeholder="Chat wiht AI ..."
            fullWidth
            sx={{
              backgroundColor: "white",
              color: "black",
            }}
            value={text.message}
            onChange={updateText}
          />
          <ButtonGroup
            variant="contained"
            aria-label="outlined primary button group"
            size="small"
          >
            <Button className="btn" onClick={handleSend}>
              Send
            </Button>
            <Button className="btn" onClick={handleGenerate}>
              Generate
            </Button>
            <Button className="btn" onClick={handleUpdate}>
              Update
            </Button>
          </ButtonGroup>
        </Box>
      </Box>
    </ChatBox>
  );
};

export default ChatAI;
