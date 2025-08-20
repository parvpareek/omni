import React, { useState, useRef, useEffect } from 'react';

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws';

function Chat() {
  const [messages, setMessages] = useState([
    { role: 'omni', text: 'Welcome! Ask your spiritual question.' }
  ]);
  const [input, setInput] = useState('');
  const [connected, setConnected] = useState(false);
  const ws = useRef(null);
  const chatEndRef = useRef(null);

  const connect = () => {
    if (ws.current && connected) return;
    const socket = new window.WebSocket(WS_URL);
    ws.current = socket;
    socket.onopen = () => setConnected(true);
    socket.onclose = () => setConnected(false);
    socket.onerror = () => setConnected(false);
    socket.onmessage = async (event) => {
      try {
        if (typeof event.data === 'string') {
          // Try JSON first
          try {
            const data = JSON.parse(event.data);
            if (data.type === 'text' && data.text) {
              setMessages((msgs) => [...msgs, { role: 'omni', text: data.text }]);
              return;
            }
          } catch (_) {
            /* not JSON */
          }
          // Treat as plain text
          setMessages((msgs) => [...msgs, { role: 'omni', text: event.data }]);
        } else if (event.data instanceof Blob) {
          const text = await event.data.text();
          setMessages((msgs) => [...msgs, { role: 'omni', text }]);
        }
      } catch (e) {
        console.warn('Message handling error', e);
      }
    };
  };

  const disconnect = () => {
    if (ws.current) {
      ws.current.close();
      ws.current = null;
    }
  };

  useEffect(() => {
    return () => {
      if (ws.current) {
        ws.current.close();
        ws.current = null;
      }
    };
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = () => {
    if (!input.trim() || !connected || !ws.current) return;
    setMessages((msgs) => [...msgs, { role: 'user', text: input }]);
    // Send raw text so FastAPIWebsocketTransport can treat it as TextFrame
    ws.current.send(input);
    setInput('');
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: 380 }}>
      <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
        <button onClick={connect} disabled={connected} style={{ padding: '6px 12px' }}>Connect</button>
        <button onClick={disconnect} disabled={!connected} style={{ padding: '6px 12px' }}>Disconnect</button>
        <span style={{ marginLeft: 8, color: connected ? 'green' : 'red' }}>{connected ? 'Connected' : 'Disconnected'}</span>
      </div>
      <div style={{ flex: 1, overflowY: 'auto', marginBottom: 12 }}>
        {messages.map((msg, i) => (
          <div key={i} style={{
            textAlign: msg.role === 'user' ? 'right' : 'left',
            margin: '8px 0',
          }}>
            <span style={{
              display: 'inline-block',
              background: msg.role === 'user' ? '#e3f2fd' : '#f3e5f5',
              color: '#333',
              borderRadius: 8,
              padding: '8px 14px',
              maxWidth: '80%',
              wordBreak: 'break-word',
            }}>
              {msg.text}
            </span>
          </div>
        ))}
        <div ref={chatEndRef} />
      </div>
      <div style={{ display: 'flex', gap: 8 }}>
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && sendMessage()}
          placeholder="Type your question..."
          style={{ flex: 1, padding: 10, borderRadius: 6, border: '1px solid #ccc' }}
          disabled={!connected}
        />
        <button
          onClick={sendMessage}
          disabled={!connected || !input.trim()}
          style={{ padding: '0 18px', borderRadius: 6, border: 'none', background: '#1976d2', color: '#fff', cursor: 'pointer' }}
        >
          Send
        </button>
      </div>
    </div>
  );
}

export default Chat; 