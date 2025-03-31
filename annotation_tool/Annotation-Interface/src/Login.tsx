import React, { useState, useEffect, useContext } from 'react';
import AuthContext from './AuthContext';

const host = process.env.REACT_APP_API_HOST || "localhost";
const port = process.env.REACT_APP_API_PORT || 8000;

function Login() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const { setUser } = useContext(AuthContext);


  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    fetch('http://'+ host+':'+port+'/login', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ username, password })
    })
    .then(response => {
      if (!response.ok) {
        throw new Error("HTTP status " + response.status);
      }
      return response.json();
    })
    .then(data => {
      console.log(data);
      setUser(username);
    })
    .catch(error => {
      if (error.message.startsWith("HTTP status ")) {
        setErrorMessage("Invalid username or password");
      } else {
        setErrorMessage("An unexpected error occurred");
      }
    });
  };

  return (
    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh', background: '#f5f5f5' }}>
      <div style={{ padding: '20px', borderRadius: '5px', background: '#fff', boxShadow: '0px 0px 10px rgba(0, 0, 0, 0.1)' }}>
        <h2 style={{ textAlign: 'center', marginBottom: '20px' }}>Login</h2>
        <form onSubmit={handleSubmit}>
          <input type="text" value={username} onChange={(e) => setUsername(e.target.value)} placeholder="Username" required style={{ display: 'block', width: '100%', padding: '10px', marginBottom: '10px', fontSize: '16px' }} />
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Password" required style={{ display: 'block', width: '100%', padding: '10px', marginBottom: '10px', fontSize: '16px' }} />
          <button type="submit" style={{ display: 'block', width: '100%', padding: '10px', marginBottom: '10px', background: '#007BFF', color: '#fff', border: 'none', borderRadius: '5px', fontSize: '16px', cursor: 'pointer' }}>Log in</button>
          {errorMessage && <p style={{ color: 'red', textAlign: 'center' }}>{errorMessage}</p>}
        </form>
        <p style={{ textAlign: 'center', color: '#888' }}>You need to create your username and password in ./backend/user.py or contact admin.</p>
      </div>
    </div>
  );
}

export default Login;
