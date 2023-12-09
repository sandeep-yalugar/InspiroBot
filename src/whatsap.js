import React, { useState, useEffect, useRef } from 'react';
import './whatsap.css'; // You can style it as per your preference
import axios from 'axios';

const WhatsAppChat = () => {
  const [messages, setMessages] = useState([
    { id: 1, text: 'I am a therapeutic chatbot. How can I assist you today?', sender: 'chatbot' }
  ]);
  let flag = 0;

  // Use useRef for persistent values across renders
  const stackRef = useRef([]);
  const convStackRef = useRef([]);
  const flagRef = useRef(0);

  const [userInput, setUserInput] = useState('');
  const [isVisible, setIsVisible] = useState(false);

  const generateStory = async () => {
    const url2 = 'http://127.0.0.1:3005/createStory';
    const reqData = {
      "context": convStackRef.current
    }
    flagRef.current = 1;
    axios.post(url2, reqData, {
      headers: {
        'Content-Type': 'application/json',
      }
    }).then(response => {
      setMessages((prev) => {
        return [...prev, { id: stackRef.current.length + Date.now(), text: response.data.result, sender: 'chatbot' }];
      });
      setIsVisible(false);
    })
  }
  const generateStoryClick = async () => {
    setIsVisible(false);
    generateStory();
  }

  const handleSend = () => {
    console.log("running handle submit");

    if (userInput.trim() !== '') {
      const newMessage = {
        id: messages.length + 1,
        text: userInput,
        sender: 'user',
      };

      setMessages([...messages, newMessage]);

      stackRef.current.push(userInput);
      console.log(stackRef.current);
      convStackRef.current.push(userInput);
      console.log("convolength", convStackRef.current.length);
      if (convStackRef.current.length >= 5 && flagRef.current===0) {
        console.log("entered if condition");
        setIsVisible(true);
      }
      setUserInput('');
      const requestData = {
        "stack": stackRef.current
      };
      const url = 'http://127.0.0.1:3005/createReply';
      axios.post(url, requestData, {
        headers: {
          'Content-Type': 'application/json',
        }
      })
        .then(response => {
          console.log('Response:', response.data);
          stackRef.current.push(response.data.result)
          setMessages((prev) => {
            return [...prev, { id: stackRef.current.length + Date.now(), text: response.data.result, sender: 'chatbot' }]
          })
        })
        .catch(error => {
          console.error('Error:', error);
        });
    }
  };

  const handleEnter  = async (e) => {
    if(e.key==="Enter") {
      handleSend();
    }
  }

  return (
    <div className="whatsapp-chat-container">
      <div className="chat-messages">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`message ${message.sender === 'user' ? 'user-message' : 'chatbot-message'}`}
          >
            {message.text}
          </div>
        ))}
      </div>
      <div>
        {
          isVisible === true ? <button onClick={generateStoryClick}>a story for you</button> : null
        }
      </div>
      <div className="chat-input">
        <input
          type="text"
          placeholder="Type a message..."
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          onKeyDown={handleEnter}
        />
        <button onClick={handleSend} >Send</button>
      </div>
    </div>
  );
};

export default WhatsAppChat;
