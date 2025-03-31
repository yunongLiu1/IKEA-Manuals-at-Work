import React, { useState, useEffect } from 'react';
import './MessageDisplay.css'; // Import your CSS file

const host = process.env.REACT_APP_API_HOST || "localhost";
const port = process.env.REACT_APP_API_PORT || 8000;

const MessageDisplay: React.FC = () => {
  const [data, setData] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>('');

  useEffect(() => {
    fetch(`http://${host}:${port}/data-stat`)
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        setData(data.data);
        setLoading(false);
      })
      .catch(error => {
        console.error('There was an error!', error);
        setError('Failed to fetch data.');
        setLoading(false);
      });
  }, []);

  if (loading) {
    return <p>Loading...</p>;
  }

  if (error) {
    return <p>Error: {error}</p>;
  }

  const formattedData = data ? data.split('\n').map((line: string, index: number) => (
    <p key={index}>{line}</p>
  )) : null;

  return (
    <div className="data-container">
      <div className="data-content">{formattedData}</div>
    </div>
  );
};

export default MessageDisplay;
